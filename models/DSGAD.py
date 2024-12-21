import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as dglF
import scipy
import sympy
import sklearn.metrics as skmetrics
from .base import BaseModel, get_thres
import datasets


def calculate_theta2(d: int) -> list[list[float]]:
    '''Calculate the polynomial coefficients of the filter'''
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2)**i * (1 - x / 2)**(d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


def poly_conv(theta: list[float], g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
    '''Polynomial convolution using filter coefficients'''

    def unnLaplacian(feat, D_invsqrt, graph):
        """ Operation Feat * D^-1/2 A D^-1/2 """
        graph.ndata['h'] = feat * D_invsqrt
        graph.update_all(dglF.copy_u('h', 'm'), dglF.sum('m', 'h'))
        return feat - graph.ndata.pop('h') * D_invsqrt

    with g.local_scope():
        D_invsqrt = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(features.device)
        h = theta[0] * features
        for k in range(1, len(theta)):
            features = unnLaplacian(features, D_invsqrt, g)
            h += theta[k] * features
    return h


class DSGAD(BaseModel):

    def __init__(
            self,
            in_nodes: int,
            in_feats: int,
            h_feats: int = 64,
            num_classes: int = 2,
            d=2,
            mix_beta: int = 2,  # Mixed beta numbers
    ):
        super().__init__()

        self.thetas = calculate_theta2(d)  # Parameters for each filter
        self.num_filters = len(self.thetas)  # Number of filters

        self.input = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
        )

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
        ) for _ in range(self.num_filters + mix_beta)])

        c = self.num_filters + mix_beta
        ks = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Conv1d(c, c, ks, stride, 'same'),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(c * h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, num_classes),
        )

        # Weights used to mix beta filtering
        self.weights = nn.Parameter(torch.randn(mix_beta, self.num_filters, in_nodes, h_feats))

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        # The weight of the hybrid filter is related to the output of the first MLP
        if self.weights.shape[0] > 0:  # If there is a hybrid filter
            X = h.unsqueeze(0).unsqueeze(0)
            X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
            weights = self.weights * X
            weights = weights.sum(dim=(2, 3))

        h = [poly_conv(theta, g, h) for theta in self.thetas]  # filter
        if self.weights.shape[0] > 0:
            mix = [sum([h[i] * w[i] for i in range(self.num_filters)]) for w in weights.softmax(1)]  # hybrid filter
            h += mix

        h = [self.fcs[i](h[i]) for i in range(len(self.fcs))]  # FC

        h = [x.unsqueeze(1) for x in h]  # Add channels to the output to facilitate overconvolution
        h = torch.cat(h, 1)  # Channel merging
        h = self.conv(h)  # convolution
        h = self.classifier(h)  # Classifiers

        return h

    @classmethod
    def trainfit(cls, args):

        # Load the graph dataset
        dataset = eval(f'datasets.{args.dataset}')()
        g: dgl.DGLGraph = dataset.g
        print(f'{args.dataset}: {g}')
        g = dataset.split(g, args.ratio)
        print(f'after split: {g}')

        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        print(f'train/val/test samples: {train_mask.sum().item()} {val_mask.sum().item()} {test_mask.sum().item()}')

        g = g.to(args.device)
        features = g.ndata['feature']  # N x L   Node features
        labels = g.ndata['label']  # N   Node labels

        # Number of Normals / Abnormalities
        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print(f'cross entropy weight: {weight}')

        # model
        model = cls(features.shape[0], features.shape[1], **args.model_config).to(args.device)

        params = []
        for name, param in model.named_parameters():
            if name == 'weights': params.append({'params': param, 'lr': 1e-1})  # large learning rates for weight
            else: params.append({'params': param, 'lr': 1e-3})  # others normal learning rates
        optimizer = torch.optim.Adam(params)

        # train
        for e in range(args.epoch):

            model.train()
            out = model(g, features)
            loss = F.cross_entropy(
                out[train_mask],
                labels[train_mask],
                weight=torch.tensor([1., weight], device=args.device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(g, features)
            probs = out.softmax(1)[:, 1].cpu()  # The probability of being predicted to be a positive class
            thres = get_thres(labels[val_mask].cpu(), probs[val_mask])  # get threshold in validation set

            probs = probs[test_mask]
            y_pred = torch.zeros_like(probs)
            y_pred[probs >= thres] = 1
            y_true = labels[test_mask].cpu()

            # Test set metrics
            auc = skmetrics.roc_auc_score(y_true, probs)
            recall = skmetrics.recall_score(y_true, y_pred)
            precision = skmetrics.precision_score(y_true, y_pred)
            f1_macro = skmetrics.f1_score(y_true, y_pred, average='macro')

            print(f'Epoch {e+1}, loss: {loss.item():.5f}')
            print('auc  recall  precision  f1_macro')
            print(f'{auc:.5f}  {recall:.5f}  {precision:.5f}  {f1_macro:.5f}')

        return model, {
            'auc': auc,
            'recall': recall,
            'precision': precision,
            'f1_macro': f1_macro,
        }


class DSGAD_hete(DSGAD):

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        if self.weights.shape[0] > 0:
            X = h.unsqueeze(0).unsqueeze(0)
            X = X.repeat(self.weights.shape[0], self.weights.shape[1], 1, 1)
            weights = self.weights * X
            weights = weights.sum(dim=(2, 3))

        h_all = []
        for relation in g.canonical_etypes:

            hh = [poly_conv(theta, g[relation], h) for theta in self.thetas]
            if self.weights.shape[0] > 0:
                mix = [sum([hh[i] * w[i] for i in range(self.num_filters)]) for w in weights.softmax(1)]
                hh += mix

            hh = [self.fcs[i](hh[i]) for i in range(len(self.fcs))]

            hh = [x.unsqueeze(1) for x in hh]
            hh = torch.cat(hh, 1)
            hh = self.conv(hh)
            hh = self.classifier(hh)

            h_all.append(hh)

        h = sum(h_all) / len(h_all)

        return h

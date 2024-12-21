import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import sklearn.metrics as skmetrics

import datasets


def get_thres(labels, probs):
    '''
    get threshold according to the best macro f1

    @param labels labels
    @param probs The probability of being predicted to be a positive class
    '''
    best_f1, best_thres = 0, 0
    for thres in np.linspace(0, 1, 101):
        preds = np.zeros_like(labels)
        preds[probs >= thres] = 1
        mf1 = skmetrics.f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thres = thres
    return best_thres


class BaseModel(nn.Module):

    @classmethod
    def trainfit(cls, args):
        '''train model'''

        # load graph dataset
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
        model = cls(features.shape[1], **args.model_config).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

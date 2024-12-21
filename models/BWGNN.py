import torch
import torch.nn as nn
import dgl
import dgl.function as dglF
import scipy
import sympy
from .base import BaseModel


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


class BWGNN(BaseModel):

    def __init__(self, in_feats: int, h_feats: int = 64, num_classes: int = 2, d=2):
        super().__init__()

        self.thetas = calculate_theta2(d)

        self.input = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(h_feats * len(self.thetas), h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, num_classes),
        )

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        # Direct splicing
        h_final = [poly_conv(theta, g, h) for theta in self.thetas]
        h_final = torch.cat(h_final, -1)

        h = self.classifier(h_final)
        return h


class BWGNN_hete(BaseModel):

    def __init__(self, in_feats: int, h_feats: int = 64, num_classes: int = 2, d=2):
        super().__init__()

        self.thetas = calculate_theta2(d)

        self.input = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
        )
        self.linear3 = nn.Linear(h_feats * len(self.thetas), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor):
        h = self.input(in_feat)

        h_all = []
        for relation in g.canonical_etypes:

            # Direct splicing
            h_final = [poly_conv(theta, g[relation], h) for theta in self.thetas]
            h_final = torch.cat(h_final, -1)
            h_final = self.linear3(h_final)
            h_all.append(h_final)

        h = torch.stack(h_all).sum(0)
        h = self.act(h)
        h = self.linear4(h)

        return h

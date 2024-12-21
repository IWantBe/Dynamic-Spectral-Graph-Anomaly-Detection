import torch
import dgl
from dgl.data import FraudAmazonDataset, FraudYelpDataset
import sklearn.model_selection as skselection
from pathlib import Path

datasets_path = Path(__file__).resolve().parent


class BaseDataset:

    def __init__(self, *args, **kwargs):
        ...

    def split(self, g: dgl.DGLGraph, ratio: list[float]) -> dgl.DGLGraph:
        '''
        split the graph by 3 parts (train, val, test)

        the split masks are store in the 3 keys 'train_mask', 'val_mask', 'test_mask' of ndata
        '''

        assert len(ratio) == 3
        assert sum(ratio) == 1
        for i in ratio:
            assert 0 <= i <= 1

        num_nodes = g.num_nodes()
        labels = g.ndata['label']

        s = ['train_mask', 'val_mask', 'test_mask']
        if 1 in ratio:
            idx = ratio.index(1)
            for i in range(3):
                if i == idx:
                    g.ndata[s[i]] = torch.ones(num_nodes).bool()
                else:
                    g.ndata[s[i]] = torch.zeros(num_nodes).bool()

        elif 0 in ratio:
            idx = ratio.index(0)
            g.ndata[s[idx]] = torch.zeros(num_nodes).bool()

            idx1, idx2 = 0, 0
            for i in range(3):
                idx1 = i
                if idx1 != idx: break
            for i in range(3):
                idx2 = i
                if idx2 != idx and idx2 != idx1: break

            indeics = list(range(num_nodes))
            x_train, x_test, _, _ = skselection.train_test_split(
                indeics,
                labels,
                stratify=labels,
                train_size=ratio[idx1],
                random_state=2,
                shuffle=True,
            )
            mask1 = torch.zeros(num_nodes).bool()
            mask2 = torch.zeros(num_nodes).bool()
            mask1[x_train] = True
            mask2[x_test] = True

            g.ndata[s[idx1]] = mask1
            g.ndata[s[idx2]] = mask2

        else:
            index = list(range(num_nodes))
            idx_train, idx_rest, _, y_rest = skselection.train_test_split(
                index,
                labels,
                stratify=labels,
                train_size=ratio[0],
                random_state=2,
                shuffle=True,
            )
            idx_valid, idx_test, _, _ = skselection.train_test_split(
                idx_rest,
                y_rest,
                stratify=y_rest,
                train_size=ratio[1] / (ratio[1] + ratio[2]),
                random_state=2,
                shuffle=True,
            )
            train_mask = torch.zeros(num_nodes).bool()
            val_mask = torch.zeros(num_nodes).bool()
            test_mask = torch.zeros(num_nodes).bool()

            train_mask[idx_train] = True
            val_mask[idx_valid] = True
            test_mask[idx_test] = True

            g.ndata[s[0]] = train_mask
            g.ndata[s[1]] = val_mask
            g.ndata[s[2]] = test_mask

        return g


class yelp(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudYelpDataset(raw_dir=str(datasets_path))[0]
        # to homogeneous graph
        self.g = dgl.to_homogeneous(self.g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class tfinance(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = dgl.load_graphs(str(datasets_path / 'tfinance'))[0][0]
        self.g.ndata['label'] = self.g.ndata['label'].argmax(1)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class amazon(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudAmazonDataset(raw_dir=str(datasets_path))[0]
        # to homogeneous graph
        self.g = dgl.to_homogeneous(self.g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class tolokers(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = dgl.load_graphs(str(datasets_path / 'tolokers'))[0][0]

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class yelp_hete(BaseDataset):
    '''Heterogeneous yelp'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudYelpDataset(raw_dir=str(datasets_path))[0]

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()


class amazon_hete(BaseDataset):
    '''Heterogeneous Amazon'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.g = FraudAmazonDataset(raw_dir=str(datasets_path))[0]

        self.g.ndata['label'] = self.g.ndata['label'].long().squeeze(-1)
        self.g.ndata['feature'] = self.g.ndata['feature'].float()

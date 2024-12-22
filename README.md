# Dynamic-Spectral-Graph-Anomaly-Detection
Dynamic Spectral Graph Anomaly Detection accepted by AAAI2025.

I would like to express my sincere gratitude to Mr. Zhang Tairui for his work on the code implementation, and also extend my heartfelt thanks to the other collaborators.

If you want to use this code, please
- install python(3.9.18), dgl(2.0.0.cu118, py39_0) and pytorch(2.2.1, py3.9_cuda11.8_cudnn8.7.0_0), and numpy, scipy, sklearn, etc.
- download tfinance and tolokers datasets, and put it into the [`datasets`](datasets/) folder (yelp and amazon are built in the dgl).
- run `python t.py --model DSGAD --run 1 --dataset [yelp, tfinance, amazon, tolokers]` in homogeneous graph, `python t.py --model DSGAD_hete --run 1 --dataset [yelp_hete, amazon_hete]` in heterogeneous graph.


Note: please refer to the original code of BWGNN published on https://github.com/squareRoot3/Rethinking-Anomaly-Detection. In this paper, we have adapted it into our framework.

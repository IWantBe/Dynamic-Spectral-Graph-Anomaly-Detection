# Dynamic-Spectral-Graph-Anomaly-Detection
Dynamic Spectral Graph Anomaly Detection is published by AAAI2025.

I would like to express my sincere gratitude to Mr. Zhang Tairui for his work on the code implementation, and also extend my heartfelt thanks to the other collaborators.

If you want to use this code, please
- install python(3.9.18), dgl(2.0.0.cu118, py39_0) and pytorch(2.2.1, py3.9_cuda11.8_cudnn8.7.0_0), and numpy, scipy, sklearn, etc.
- download tfinance and tolokers datasets, and put it into the [`datasets`](datasets/) folder (yelp and amazon are built in the dgl).
- run `python t.py --model DSGAD --run 1 --dataset [yelp, tfinance, amazon, tolokers]` in homogeneous graph, `python t.py --model DSGAD_hete --run 1 --dataset [yelp_hete, amazon_hete]` in heterogeneous graph.


Note: Please refer to the original BWGNN code published at https://github.com/squareRoot3/Rethinking-Anomaly-Detection. In this paper, we have adapted it into our framework.


Citation Format

Zheng, J., Yang, C., Zhang, T., Cao, L., Jiang, B., Fan, X., Wu, X.- ming, & Zhu, X. (2025). Dynamic Spectral Graph Anomaly Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(12), 13410-13418. https://doi.org/10.1609/aaai.v39i12.33464.

@article{Zheng2025, 
  title={Dynamic Spectral Graph Anomaly Detection}, 
  volume={39}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/33464}, 
  DOI={10.1609/aaai.v39i12.33464}, 
  number={12}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Zheng, Jianbo and Yang, Chao and Zhang, Tairui and Cao, Longbing and Jiang, Bin and Fan, Xuhui and Wu, Xiao-ming and Zhu, Xianxun}, 
  year={2025}, 
  month={Apr.}, 
  pages={13410-13418} }


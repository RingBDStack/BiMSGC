# **Bi-Directional Multi-Scale Graph Dataset Condensation via Information Bottleneck（BiMSGC）**



This is the official code for AAAI 2025 Bi-Directional Multi-Scale Graph Dataset Condensation via Information Bottleneck

### Requirements
```
deeprobust==0.2.9
gdown==4.7.3
networkx==3.2.1
numpy==1.26.3
ogb==1.3.6
pandas==2.1.4
scikit-learn==1.3.2
scipy==1.11.4
torch==2.1.2
torch_geometric==2.4.0
torch-sparse==0.6.18
```

## Download Datasets
For Citeseer Pubmed and Squirrel, the code will directly download them.
For Reddit, Flickr, and Ogbn-arXiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (the links are provided by GraphSAINT team). 
Download the files and unzip them to `data` at the root directory. 

## Instructions

(1) Run preprocess.py to preprocess the dataset and conduct the spectral decomposition.

(2) Initialize node features of the synthetic graph by running feat_init.py.

(3) Distill the synthetic graph by running distill.py.

## Cite
Welcome to kindly cite our work!
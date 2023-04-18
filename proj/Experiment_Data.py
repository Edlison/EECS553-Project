import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import Net_GAT, Net_GCN, NetAmazon_GAT, NetAmazon_GCN, MyNet, Net_imp
from dataset import Amazon
import train as tr

if __name__ == '__main__':
    # args = parser.parse_args()
    # tr.train(dataset_name='cora', model_name='GAT', iterations=args.epochs)
    a = tr.train(dataset_name='cora', model_name='GAT', iterations=100)
    print(type(a))

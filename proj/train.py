# @Author  : Edlison
# @Date    : 3/5/23 23:03
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import Net_GAT, Net_GCN, NetAmazon_GAT, NetAmazon_GCN, MyNet, Net_imp
from dataset import Amazon


cuda = True if torch.cuda.is_available() else False


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora", help='enter your dataset')
parser.add_argument('--k', type=int, default=2, help='sigmoid k value')


def train(dataset_name='cora', model_name='GCN', iterations=100, lr=0.005, reg=5e-4):
    dataset = {
        'amazon': Amazon(),
        'cora': Planetoid(root='../data', name='cora'),
        'CiteSeer': Planetoid(root='../data', name='CiteSeer'),
        'PubMed': Planetoid(root='../data', name='PubMed')
    }
    dataset = dataset[dataset_name]
    if dataset_name == 'amazon':
        data = dataset
        net = {'GCN': NetAmazon_GCN(dataset.num_node_features, dataset.num_classes),
               'GAT': NetAmazon_GAT(dataset.num_node_features, dataset.num_classes)}
    else:
        data = dataset[0]
        net = {'GCN': Net_GCN(dataset.num_node_features, dataset.num_classes),
               'GAT': Net_GAT(dataset.num_node_features, dataset.num_classes)}
    model = net[model_name]
    x, edge_index = data.x, data.edge_index
    if cuda:
        device = 1
        x.cuda(device)
        edge_index(device)
        model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(x, edge_index).argmax(dim=1)
        cor_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc_val = cor_val / data.test_mask.sum()
        cor_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc_test = cor_test / data.test_mask.sum()
        print(
            'epoch: {}, loss: {:.4f}, eval acc: {:.4f}, test acc: {:.4f}'.format(epoch, loss.item(), acc_val, acc_test))

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)
    cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = cor / data.test_mask.sum()
    print('Final acc: {:.4f}'.format(acc.item()))


def train_my(iterations=100, lr=0.005, reg=5e-4):
    dataset = Planetoid(root='../data', name='cora')
    data = dataset[0]
    model = Net_imp(dataset.num_node_features, dataset.num_classes)
    x, edge_index = data.x, data.edge_index
    if cuda:
        device = 1
        x.cuda(device)
        edge_index(device)
        model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(x, edge_index).argmax(dim=1)
        cor_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc_val = cor_val / data.test_mask.sum()
        cor_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc_test = cor_test / data.test_mask.sum()
        print('epoch: {}, loss: {:.4f}, eval acc: {:.4f}, test acc: {:.4f}'.format(epoch, loss.item(), acc_val, acc_test))

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)
    cor = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = cor / data.test_mask.sum()
    print('acc: {:.4f}'.format(acc.item()))


if __name__ == '__main__':
    """
    dataset: {'cora', 'CiteSeer', 'PubMed'}
    model: {'GCN', 'GAT'}
    """
    # train(dataset_name='cora', model_name='GAT', iterations=200)
    # todo novel GAT method()
    train_my(iterations=50)

    args = parser.parse_args()

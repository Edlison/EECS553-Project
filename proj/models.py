# @Author  : Edlison
# @Date    : 3/5/23 18:49
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree


class MyGAT(MessagePassing):
    def __init__(self, in_features, out_features):
        super(MyGAT, self).__init__(aggr='mean', flow='source_to_target', node_dim=-2)
        self.lin = torch.nn.Linear(in_features, out_features)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.softmax = torch.nn.Softmax(dim=0)
        self.a = Parameter(torch.zeros([2 * out_features, 1]))
        torch.nn.init.xavier_uniform_(self.a, gain=1.414)
        # self.e_all = Parameter(torch.zeros([in_features]), requires_grad=False)

    def attn(self, h, edge_index):
        row, col = edge_index  # 分别拿到src_index, tgt_index. shape是[10556+2708(self_loop)]
        a_in = torch.cat([h[row], h[col]], dim=1)  # 获得Wh_i和Wh_j拼接 [13264, 2*out_features]
        e = torch.mm(a_in, self.a).squeeze()  # [13264]
        e = self.leaky_relu(e)

        # if cuda check device for {e_all, norm}
        e_all = torch.zeros(h.shape[0])  # [2708]
        for i, tgt in enumerate(col):
            e_all[tgt] += torch.exp(e[i])

        # calculates e
        norm = torch.zeros(e.shape[0])  # [13264]
        for i, item in enumerate(e):
            norm[i] = torch.exp(item) / e_all[col[i]]

        # F.batch_norm()
        return norm

    def forward(self, x, edge_index):  # x: [2708, 1433], edge_index: [2, 10556]
        h = self.lin(x)  # 先通过一个线性变换 x[2708, 1433]->h[2708, out_features]
        print(h.shape)
        edge_index, _ = add_self_loops(edge_index)
        norm = self.attn(h, edge_index)
        out = self.propagate(edge_index, x=h, norm=norm)  # out: [2708, 16], h: [2708, 16], norm: [13264]

        return out

    def message(self, x_j, norm):  # x_j 是自动获得的所有邻居节点的信息
        print('xj: ', x_j.shape)
        return norm.view(-1, 1) * x_j  # norm: [13264, 1], x_j: [13264, 16]


class MyNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = MyGAT(num_node_features, 64)
        self.gat2 = MyGAT(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)


class Net_imp(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 8, heads=8)
        # self.lstm = torch.nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        self.batchnorm = torch.nn.BatchNorm1d(64)
        self.gat2 = GATConv(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        # h, _ = self.lstm(h.view(1, h.shape[0], h.shape[1]))
        # h = self.batchnorm(h)
        h = self.gat2(h, edge_index)

        return F.log_softmax(h, dim=1)


# Transduction
class Net_GAT(torch.nn.Module):
    """
    Two-layer GAT
    Heads = 8
    Features per head = 8
    1st layer activator: exponential linear unit (ELU)
    2nd layer activator: softmax
    L-2 regularization = 0.0005
    Dropout = 0.6
    """
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(64, num_classes, dropout=0.6)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.gat2(h, edge_index)

        return F.log_softmax(h, dim=1)


class Net_GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, 8)
        self.gcn2 = GCNConv(8, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.gcn2(h, edge_index)

        return F.log_softmax(h, dim=1)


class NetAmazon_GAT(torch.nn.Module):
    # todo add layers, without dropout, change activator
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=16)
        self.gat2 = GATConv(256, 8, heads=8)
        self.gat3 = GATConv(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.gat3(h, edge_index)

        return F.log_softmax(h, dim=1)


class NetAmazon_GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, 256)
        self.gcn2 = GCNConv(256, 64)
        self.gcn3 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.gcn2(h, edge_index)
        h = F.relu(h)
        h = self.gcn3(h, edge_index)

        return F.log_softmax(h, dim=1)
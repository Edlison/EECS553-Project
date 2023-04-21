# @Author  : Edlison
# @Date    : 3/5/23 18:49
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
# from torch.nn.init import xavier_uniform_
from torch_geometric.nn import MessagePassing, GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax


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


class MyMHGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super(MyMHGAT, self).__init__(aggr='mean', flow='source_to_target', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = 0.2
        self.lin = Linear(in_channels, heads * out_channels)
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self._ini_parameters()

    def _ini_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        alpha = (x * self.att).sum(dim=-1)
        # print('alpha: ', alpha.shape, 'x: ', x.shape, 'att: ', self.att.shape)

        edge_index, _ = add_self_loops(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out.view(-1, self.heads * self.out_channels) + self.bias

        return out

    def message(self, x_j, alpha_j, index):
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # print('alpha', alpha.shape, 'index', index.shape)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class MyNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = MyMHGAT(num_node_features, 16, heads=8, dropout=0.6)
        self.gat2 = MyMHGAT(128, 8, heads=8, dropout=0.6)
        self.gat3 = MyMHGAT(64, num_classes, dropout=0.6)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.gat2(h, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.gat3(h, edge_index)

        return F.log_softmax(h, dim=1)


class Net_imp(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(64, 4, heads=8, dropout=0.6)
        self.attention = torch.nn.MultiheadAttention(32, num_heads=8, dropout=0.6)
        self.fc = Linear(32, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h.unsqueeze_(0)
        h, weight = self.attention(h, h, h)
        # h = F.layer_norm(h, [h.shape[-2], h.shape[-1]])
        h.squeeze_()
        h = self.fc(h)
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
        self.gcn1 = GCNConv(num_node_features, 64)
        self.gcn2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.gcn2(h, edge_index)

        return F.log_softmax(h, dim=1)


class NetAmazon_GAT(torch.nn.Module):
    # todo add layers, without dropout, change activator
    def __init__(self, num_node_features, num_classes, heads=8, dropout=0.0):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=heads, dropout=dropout)
        self.gat2 = GATConv(128, 8, heads=8, dropout=dropout)
        self.gat3 = GATConv(64, num_classes, dropout=dropout)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = self.gat3(h, edge_index)

        return F.log_softmax(h, dim=1)


class NetAmazon_GAT_heads(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads, dropout=0.0):
        super().__init__()
        # original hide parameter gat1: 16, gat2: 8
        self.gat1 = GATConv(num_node_features, 128, heads=heads, dropout=dropout)
        self.gat2 = GATConv(128 * heads, 64, heads=heads)
        self.gat3 = GATConv(64 * heads, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.gat3(h, edge_index)
        return F.log_softmax(h, dim=1)


class NetAmazon_GAT_layers_2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads, dropout=0.0):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=heads, dropout=dropout)
        self.gat2 = GATConv(16 * heads, 8, heads=heads)
        self.gat3 = GATConv(8 * heads, 8, heads=heads)
        self.gat4 = GATConv(8 * heads, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)


class NetAmazon_GAT_layers_4(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads, dropout=0.0):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 16, heads=heads, dropout=dropout)
        self.gat2 = GATConv(16 * heads, 8, heads=heads)
        self.gat3 = GATConv(8 * heads, 8, heads=heads)
        self.gat4 = GATConv(8 * heads, num_classes)

    def forward(self, x, edge_index):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.gat3(h, edge_index)
        h = F.relu(h)
        h = self.gat4(h, edge_index)
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
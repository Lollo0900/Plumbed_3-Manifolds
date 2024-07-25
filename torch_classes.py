from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch.nn import Linear, LeakyReLU , Sequential
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MLP, GCNConv, NNConv, MessagePassing, BatchNorm
from torch_scatter import scatter_mean


class GraphData(Data):
    def __init__(self, edge_index=None, x=None, edge_attr= None,y=None):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x = x
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class GraphPair(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_attr_s= None,
                 edge_index_t=None, x_t=None, edge_attr_t= None,y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphAggregator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAggregator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self.in_channels, self.out_channels))
        layer.append(nn.PReLU())
        self.mlp = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self.in_channels, self.out_channels))
        self.mlp_gate = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self.out_channels, self.hidden_channels))
        layer.append(nn.LeakyReLU())
        layer.append(nn.Linear(self.hidden_channels, self.out_channels))
        self.mlp_final = nn.Sequential(*layer)

    def forward(self, x):
        x_states = self.mlp(x)
        x_gates = F.softmax(self.mlp_gate(x), dim=1)
        x_states = x_states * x_gates
        x_states = self.mlp_final(x_states)

        return x_states


class GCNGAT(torch.nn.Module):
    def __init__(self):
        super(GCNGAT, self).__init__()
        self.gcnconv = GCNConv(in_channels=3, out_channels=64)
        self.gatconv = GATConv(in_channels=64, out_channels=64, edge_dim=1)

        self.aggregator = GraphAggregator(in_channels=64, hidden_channels=48, out_channels=32)
        self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

    def compute_embedding(self, x, edge_index, x_batch, edge_attr):
        x = self.gcnconv(x, edge_index)
        x = self.gatconv(x, edge_index, edge_attr)
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim=0)

        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out


class GCNGCN(torch.nn.Module):
    def __init__(self):
        super(GCNGCN, self).__init__()
        self.gcnconv = GCNConv(in_channels=3, out_channels=64)
        self.gcnconv1 = GCNConv(in_channels=64, out_channels=64)

        self.aggregator = GraphAggregator(in_channels=64, hidden_channels=48, out_channels=32)
        self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

    def compute_embedding(self, x, edge_index, x_batch, edge_attr):
        x = self.gcnconv(x, edge_index)
        x = self.gcnconv1(x, edge_index)
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim=0)

        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out


class NNConvGCN(torch.nn.Module):
    def __init__(self):
        super(NNConvGCN, self).__init__()
        self.nnconv1 = NNConv(in_channels=3, out_channels=64,
                              nn=Sequential(Linear(1, 32), LeakyReLU(0.01), Linear(32, 3*64)), aggr='mean')
        self.gcnconv1 = GCNConv(in_channels=64, out_channels=64)

        self.aggregator = GraphAggregator(in_channels=64, hidden_channels=48, out_channels=32)
        self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

    def compute_embedding(self, x, edge_index, x_batch, edge_attr):
        x = self.nnconv1(x, edge_index, edge_attr)
        x = self.gcnconv1(x, edge_index)
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim=0)

        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out

class NNConvNNConv(torch.nn.Module):
    def __init__(self):
        super(NNConvNNConv, self).__init__()
        self.nnconv1 = NNConv(in_channels=3, out_channels=64,
                              nn=Sequential(Linear(1, 32), LeakyReLU(0.01), Linear(32, 3*64)), aggr='mean')
        self.nnconv2 = NNConv(in_channels=64, out_channels=3,
                              nn=Sequential(Linear(1, 32), LeakyReLU(0.01), Linear(32, 64*3)), aggr='mean')
        self.aggregator = GraphAggregator(in_channels= 3, hidden_channels=48, out_channels=32)
        self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

    def compute_embedding(self, x, edge_index, x_batch, edge_attr):
        x = self.nnconv1(x, edge_index, edge_attr)
        x = self.nnconv2(x, edge_index,edge_attr)
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim=0)

        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out

class NNConvGAT(torch.nn.Module):
        def __init__(self):
            super(NNConvGAT, self).__init__()
            self.nnconv = NNConv(in_channels=3, out_channels=64,
                                  nn=Sequential(Linear(1, 32), LeakyReLU(0.01), Linear(32, 3 * 64)), aggr='mean')
            self.gatconv = GATConv(in_channels=64, out_channels=64, edge_dim=1)

            self.aggregator = GraphAggregator(in_channels=64, hidden_channels=48, out_channels=32)
            self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

        def compute_embedding(self, x, edge_index, x_batch, edge_attr):
            x = self.nnconv(x, edge_index, edge_attr)
            x = self.gatconv(x, edge_index, edge_attr)
            x = self.aggregator(x)
            x = scatter_mean(x, x_batch, dim=0)

            return x

        def forward(self, data):
            x_s, x_t = data.x_s, data.x_t
            edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
            edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
            x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
            embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
            embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
            out = torch.cat((embed_s, embed_t), 1)
            out = out.reshape(out.size(0), -1)
            out = self.classification(out)
            return out


class GEmbedLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, hidden_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.in_chs = in_channels
        self.ed_chs = edge_channels
        self.hid_chs = hidden_channels
        self.out_chs = out_channels
        self._build_model()
        self.batch_norm = BatchNorm(self.out_chs)

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self.in_chs, self.out_chs))
        layer.append(nn.LeakyReLU())
        self.mlp_node = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self.ed_chs, self.out_chs))
        layer.append(nn.LeakyReLU())
        self.mlp_edges = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(3 * self.out_chs, self.hid_chs))
        layer.append(nn.LeakyReLU())
        layer.append(nn.Linear(self.hid_chs, self.out_chs))
        self.mlp_msg = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self.in_chs + self.out_chs, self.out_chs))
        layer.append(nn.LeakyReLU())
        self.mlp_upd = nn.Sequential(*layer)

    def forward(self, x, edge_index, edge_attr):
        x_encoded = self.mlp_node(x)
        edge_encoded = self.mlp_edges(edge_attr)
        return self.propagate(edge_index, x=x_encoded, edge_attr=edge_encoded, x_original=x, edge_original=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j,edge_attr], dim=1)

        return self.mlp_msg(tmp)

    def update(self, aggr_out, x, edge_attr, x_original, edge_original):
        temp = torch.cat((x_original, aggr_out), dim=1)
        aggr_out = self.mlp_upd(temp)
        aggr_out = self.batch_norm(aggr_out)
        return aggr_out

class GENGAT(torch.nn.Module):
        def __init__(self):
            super(GENGAT, self).__init__()
            self.gembed = GEmbedLayer(in_channels=3, edge_channels=1, hidden_channels=32, out_channels=64)
            self.gatconv = GATConv(in_channels=64, out_channels=64, edge_dim=1)

            self.aggregator = GraphAggregator(in_channels=64, hidden_channels=48, out_channels=32)
            self.classification = MLP(in_channels=64, hidden_channels=32, out_channels=2, num_layers=3)

        def compute_embedding(self, x, edge_index, x_batch, edge_attr):
            x = self.gembed(x, edge_index, edge_attr)
            x = self.gatconv(x, edge_index, edge_attr)
            x = self.aggregator(x)
            x = scatter_mean(x, x_batch, dim=0)

            return x

        def forward(self, data):
            x_s, x_t = data.x_s, data.x_t
            edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
            edge_attr_s, edge_attr_t = data.edge_attr_s, data.edge_attr_t
            x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
            embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch, edge_attr_s)
            embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch, edge_attr_t)
            out = torch.cat((embed_s, embed_t), 1)
            out = out.reshape(out.size(0), -1)
            out = self.classification(out)
            return out
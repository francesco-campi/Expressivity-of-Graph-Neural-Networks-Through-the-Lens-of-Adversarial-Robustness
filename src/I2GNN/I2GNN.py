import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import NNConv, GCNConv, RGCNConv, GatedGraphConv, GINConv
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import (
    global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch, degree

import numpy as np


class I2GNN(torch.nn.Module):
    def __init__(self, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, y_ndim=1, **kwargs):
        super(I2GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI
        self.y_ndim = y_ndim
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(2, 8)

        self.z_embedding_list = torch.nn.ModuleList()
        self.transform_before_conv = torch.nn.ModuleList()
        # self.z_embedding = torch.nn.Embedding(1000, 64)
        # self.node_type_embedding = torch.nn.Embedding(100, 8)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = 64, 64
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(GatedGraphConv(M_out, 1))
        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        self.transform_before_conv.append(Linear(2*M_in, M_in))
        # self.convs.append(GraphConv(M_in, M_out))
        # self.convs.append(GatedRGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))

        for i in range(num_layers - 1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))
            # self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
            self.convs.append(GatedGraphConv(M_out, 1))
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            self.transform_before_conv.append(Linear(2 * M_in, M_in))
            # self.convs.append(GINConv(nn))
            # self.convs.append(GraphConv(M_in, M_out))

        # subgraph gnn
        # self.subgraph_convs = torch.nn.ModuleList()
        # for i in range(2):
        #    M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
       #     nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
       #     self.subgraph_convs.append(NNConv(M_in, M_out, nn))

        # MLPs for hierarchical pooling
        # self.tuple_pooling_nn = Sequential(Linear(2 * M_out, M_out), ReLU(), Linear(M_out, M_out))
        self.edge_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))
        self.node_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))
        
        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

        # self.final_mlp = Sequential(self.fc1,
        #                             torch.nn.ELU(),
        #                             self.fc2,
        #                             torch.nn.ELU(),
        #                             self.fc3)

    def forward(self, data):

        # node label embedding
        # z_emb = 0
        # if 'z' in data:
            ### computing input node embedding
       #     z_emb = self.z_embedding(data.z)
       #     if z_emb.ndim == 3:
        #        z_emb = z_emb.sum(dim=1)

        # if self.use_rd and 'rd' in data:
        #     rd_proj = self.rd_projection(data.rd)
        #     z_emb += rd_proj

        # integer node type embedding
        # x = z_emb
        x = None

        # concatenate with continuous node features
        # x = torch.cat([x, data.x.view(-1, 1)], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)

        for layer, conv in enumerate(self.convs):
            z = self.z_embedding_list[layer](data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            if x == None:
                x = z.clone()
            else:
                x = torch.cat([x, z], dim=-1)
                x = self.transform_before_conv[layer](x)
            # x = F.elu(conv(x, data.edge_index, data.edge_attr))
            # x = x + global_add_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]
            x = conv(x, data.edge_index)
            # x = x + center_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]
            # x = x + global_add_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]

        # inner-subgraph2 communications
        # x_e = global_add_pool(x, data.node_to_subgraph2) # first learn (i,j)
        # x_e = x_e[data.node_to_subgraph2] # spread to all nodes
        # x = torch.cat([x, x_e], dim=-1)
        # x = torch.cat([x, center_pool(x, data.node_to_subgraph2)[data.node_to_subgraph2]], dim=-1)
        # x = self.tuple_pooling_nn(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_add_pool(x, data.node_to_subgraph2)
            x = self.edge_pooling_nn(x)
            # x_e = global_add_pool(x, data.node_to_subgraph2)
            # x = torch.cat([x, x_e], dim=-1)
            # x = self.edge_pooling_nn(x)
            # x = global_add_pool(x, data.node_to_subgraph2)
            x = global_add_pool(x, data.subgraph2_to_subgraph)
            x = self.node_pooling_nn(x)
            # x = global_add_pool(x, data.node_to_subgraph_node)
            # x = self.edge_pooling_nn(x)
            # x = global_add_pool(x, data.subgraph_node_to_subgraph)
            # x = self.node_pooling_nn(x)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph2 = data.node_to_subgraph2.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph2, return_index=True)
            x = x[center_indices]
            x = global_mean_pool(x, data.subgraph2_to_subgraph)

        # pooling to subgraph node
        # x = scatter_mean(x, data.node_to_subgraph_node, dim=0)
        # subgraph gnn
        # x = x
        # for conv in self.subgraph_convs:
        #    x = F.elu(conv(x, data.subgraph_edge_index, data.subgraph_edge_attr))
        # subgraph node to subgraph
        # x = scatter_mean(x, data.subgraph_node_to_subgraph, dim=0)
        # subgraph to graph
        if self.y_ndim == 1:
            x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        # x = self.final_mlp(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x
"""Message passing neural network"""

from torch import nn
import torch
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool, GINConv
import torch_geometric.data
from typing import Callable
import torch_geometric


class GCN(torch_geometric.nn.models.GCN):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        num_layers: int, 
        out_channels: int, 
        act: Callable = nn.ReLU(), 
        normalize: bool = False, 
        pooling = 'add', 
        dropout = 0.0, 
        jk = None, 
        aggr = 'add'
    ) -> None:
    
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, act=act, dropout = dropout, jk = jk, aggr=aggr, normalize= normalize)
        self.final_act = nn.ReLU()
        self.final_mlp = nn.Linear(in_features=out_channels, out_features=1)
        self.pooling = pooling
    
    def forward(self, data: torch_geometric.data.Data):
        x = super().forward(data.x, data.edge_index)
        # readout function
        if self.pooling == 'add':
            x = global_add_pool(x, data.batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, data.batch)
        count = self.final_mlp(x)
        #count = self.final_act(count)
        return count

    
class GIN(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        num_layers: int,
        out_channels: int, 
        act = 'relu', 
        pooling = 'add', 
        dropout = 0.0, 
        concat = False, 
        aggr = 'add',
        batch_norm = False,
    ) -> None:

        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self. aggr = aggr
        self.convs = nn.ModuleList()

        # initialize the activation function
        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'softmax':
            self.act = torch.nn.Softmax()
        elif act == 'tanh':
            self.act = torch.nn.Tanh()
        elif act == 'elu':
            self.act = torch.nn.ELU()
        else:
            raise ValueError("Activation function not supported!")


        # first conv
        self.convs.append(self.create_conv(in_channels, 2*hidden_channels, hidden_channels))

        for i in range(num_layers-2):
            self.convs.append(self.create_conv(hidden_channels, 2*hidden_channels, hidden_channels))
        
        #   last conv
        self.convs.append(self.create_conv(hidden_channels, 2*hidden_channels, out_channels))
        
        # readout funciton
        if concat:
            self.final_mlp = nn.Sequential(
                nn.Linear(hidden_channels*(num_layers-1) + out_channels, out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        
        self.pooling = pooling
        self.concat = concat

    def forward(self, data: torch_geometric.data.Data):
        
        x = data.x
        edge_index = data.edge_index
        x_concat = []
        # apply convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.concat:
                x_concat.append(x)
        
        if self.concat:
            x = torch.cat(x_concat, 1)

        # readout function
        if self.pooling == 'add':
            x = global_add_pool(x, data.batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, data.batch)
        
        count = self.final_mlp(x)
  
        return count

    def create_conv(self, in_channels, hidden_channels, out_channels):
        mlp = [nn.Linear(in_channels, hidden_channels), self.act]

        if self.batch_norm:
            mlp.append(nn.BatchNorm1d(hidden_channels))
        if self.dropout > 0:
            mlp.append(nn.Dropout(self.dropout))

        mlp.extend([nn.Linear(hidden_channels, out_channels), self.act])

        if self.batch_norm:
            mlp.append(nn.BatchNorm1d(out_channels))
        if self.dropout > 0:
            mlp.append(nn.Dropout(self.dropout))

        mlp = nn.Sequential(*mlp)
        return GINConv(mlp, 0, True, aggr=self.aggr)



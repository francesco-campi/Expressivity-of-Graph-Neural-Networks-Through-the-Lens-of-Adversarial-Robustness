import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch


class PPGN(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,  
        out_channels: int,
        num_layers: int,  
        depth_of_mlp: int,
        pooling: str = 'add', 
        concat: bool = True,
        act: str = 'relu',
        ):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.concat = concat
        self.pooling = pooling

        # initialize the activation function
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'softmax':
            self.act = nn.Softmax()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Activation function not supported!")

        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList()
        # first block
        mlp_block = RegularBlock(in_channels + 1, hidden_channels, depth_of_mlp) # first layer has also the adjecency matrix
        self.reg_blocks.append(mlp_block)
        for _ in range(num_layers - 2):
            mlp_block = RegularBlock(hidden_channels, hidden_channels, depth_of_mlp)
            self.reg_blocks.append(mlp_block)
        #last block
        mlp_block = RegularBlock(hidden_channels, out_channels, depth_of_mlp)
        self.reg_blocks.append(mlp_block)

        # Second part
        if concat:
            self.final_mlp = nn.Sequential(
                nn.Linear(2*(hidden_channels*(num_layers-1) + out_channels), out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(2*out_channels, out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        

    def forward(self, data: Data):

        # generate the input tensor
        device = data.edge_index.device
        nodes_feat, nodes_mask = to_dense_batch(data.x, data.batch)
        adj = torch.unsqueeze(to_dense_adj(data.edge_index, data.batch, max_num_nodes=nodes_feat.shape[1]),1).to(device) # b * 1 * n * n
        # put the node features on the diagonal
        shapes = [nodes_feat.shape[0], nodes_feat.shape[2], nodes_feat.shape[1], nodes_feat.shape[1]]
        diag_nodes_feat = torch.empty(*shapes).to(device) # b*c*n*n
        for b in range(nodes_feat.shape[0]): # along the batches
            for f in range(nodes_feat.shape[-1]):
                diag_nodes_feat[b, f, :, :] = torch.diag(nodes_feat[b,:,f])
        # concatenate the two matrices
        x = torch.cat((diag_nodes_feat, adj), dim=1)
        
        # run the model
        x_concat = []
        for block in self.reg_blocks:
            x = block(x)
            if self.concat:
                x_concat.append(x)
        if self.concat:
            x = torch.cat(x_concat, 1)

        # readout function
        if self.pooling == 'add':
            x = diag_offdiag_sum_pool(x)
        elif self.pooling == 'mean':
            pass #x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = diag_offdiag_maxpool(x)
        
        count = self.final_mlp(x)

        return count


class PPGNexpl(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,  
        out_channels: int,
        num_layers: int,  
        depth_of_mlp: int,
        pooling: str = 'add', 
        concat: bool = True,
        act: str = 'relu',
        ):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.concat = concat
        self.pooling = pooling

        # initialize the activation function
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'softmax':
            self.act = nn.Softmax()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Activation function not supported!")

        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList()
        # first block
        mlp_block = RegularBlock(in_channels + 1, hidden_channels, depth_of_mlp) # first layer has also the adjecency matrix
        self.reg_blocks.append(mlp_block)
        for _ in range(num_layers - 2):
            mlp_block = RegularBlock(hidden_channels, hidden_channels, depth_of_mlp)
            self.reg_blocks.append(mlp_block)
        #last block
        mlp_block = RegularBlock(hidden_channels, out_channels, depth_of_mlp)
        self.reg_blocks.append(mlp_block)

        # Second part
        if concat:
            self.final_mlp = nn.Sequential(
                nn.Linear(2*(hidden_channels*(num_layers-1) + out_channels), out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(2*out_channels, out_channels), 
                self.act, 
                nn.Linear(out_channels, 1)
            )
        

    def forward(self, data: Data):

        # generate the input tensor
        device = data.edge_index.device
        nodes_feat, nodes_mask = to_dense_batch(data.x, data.batch)
        self.adj = torch.unsqueeze(to_dense_adj(data.edge_index, data.batch, max_num_nodes=nodes_feat.shape[1]),1).to(device) # b * 1 * n * n
        self.adj.type('torch.DoubleTensor')
        self.adj.requires_grad = True
        # put the node features on the diagonal
        shapes = [nodes_feat.shape[0], nodes_feat.shape[2], nodes_feat.shape[1], nodes_feat.shape[1]]
        diag_nodes_feat = torch.empty(*shapes).to(device) # b*c*n*n
        for b in range(nodes_feat.shape[0]): # along the batches
            for f in range(nodes_feat.shape[-1]):
                diag_nodes_feat[b, f, :, :] = torch.diag(nodes_feat[b,:,f])
        # concatenate the two matrices
        x = torch.cat((diag_nodes_feat, self.adj), dim=1)
        
        # run the model
        x_concat = []
        for block in self.reg_blocks:
            x = block(x)
            if self.concat:
                x_concat.append(x)
        if self.concat:
            x = torch.cat(x_concat, 1)

        # readout function
        if self.pooling == 'add':
            x = diag_offdiag_sum_pool(x)
        elif self.pooling == 'mean':
            pass #x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = diag_offdiag_maxpool(x)
        
        count = self.final_mlp(x)

        return count

def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S

# add also a sum and mean pool
def diag_offdiag_sum_pool(input):
    N = input.shape[-1] # number of nodes

    sum_diag = torch.sum(torch.diagonal(input, dim1=-2, dim2=-1), dim = 2)
    global_sum = torch.sum(input, dim=(-2,-1))
    sum_offdiag = global_sum - sum_diag

    return torch.cat((sum_diag, sum_offdiag), dim=1)


class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, in_features: int, out_features :int, depth_of_mlp: int):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)

        self.skip = SkipConnection(in_features+out_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
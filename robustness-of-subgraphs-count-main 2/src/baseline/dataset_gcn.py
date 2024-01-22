"""Dataset class for the message passing neural network"""

import os
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as Dataset_geometric
from torch_geometric.data import Data
import torch
import dgl
from typing import Callable


"""class GCNDataset(Dataset):

    def __init__(self, dataset_path: str, task: str, normalize: bool = True) -> None:
        super().__init__()
        graphs, labels = dgl.load_graphs(dataset_path)
        self.labels = labels[task]
        assert len(graphs) == len(self.labels), "Number fo graphs and number of labels don't match."
        self.H = []
        self.A_tilde = []
        for graph in graphs:
            self.H.append(torch.ones((graph.num_nodes(), 1), dtype=torch.float32))
            A = graph.adj() + torch.eye(graph.num_nodes(), graph.num_nodes()).to_sparse()
            if normalize:
                D = torch.diag(graph.in_degrees()).to(dtype=torch.float32).to_sparse()
                D_sqrt = torch.pow(D, -0.5)
                A = torch.matmul(torch.matmul(D_sqrt, A), D_sqrt)
            self.A_tilde.append(A)

    
    def __len__(self):
        return len(self.labels)
        
    
    def __getitem__(self, index):
        return self.H[index], self.A_tilde[index].to_dense(), self.labels[index]"""


class GraphDataset(Dataset):

    def __init__(self, dataset_path: str, task: str, in_channels: int = 0):#,  transform: Callable = None, pre_transform: Callable = None, pre_filter: Callable = None):
        super().__init__()
        self.task = task
        self.dataset_path = dataset_path
        self.in_channels = in_channels
        # process the data
        self._process()

    def _process(self):
        graphs, labels = dgl.load_graphs(self.dataset_path)
        self.labels: torch.Tensor = labels[self.task]
        assert len(graphs) == len(self.labels), f"Number fo graphs and number of labels don't match: number of graphs is {len(graphs)} while number of labels is {len(self.labels)}"
        self.data = []
        for graph, count in zip(graphs, self.labels):
            x = torch.ones((graph.num_nodes(), self.in_channels), dtype=torch.float32)
            y = torch.tensor([[count]])
            edge_index = torch.stack(graph.edges(), dim = 0)
            self.data.append(Data(x, edge_index, y = y).coalesce())
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]

    def get_statistics(self):
        variance = self.labels.var()
        standard_deviation = self.labels.std()
        mean = self.labels.mean()
        return mean, variance, standard_deviation

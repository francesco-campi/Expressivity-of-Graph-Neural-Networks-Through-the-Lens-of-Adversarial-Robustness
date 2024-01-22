import os
import os.path as osp
import sys
from typing import Callable, List, Optional, Tuple, Dict
import copy
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from dgl import load_graphs
import networkx as nx
from torch_geometric.data import Data

from .batch import Batch  # replace with custom Batch to handle subgraphs
import collections.abc as container_abcs

int_classes = int
string_classes = (str, bytes)


class I2GNNDataset(InMemoryDataset):
    def __init__(self, dataset: str, root: str, subgraph_type : str, hops: int, processed_name='processed',
                 transform=None, pre_transform=None, pre_filter=None, url=None,):
        self.url = url
        self.hops = hops
        self.root = root
        self.dataset = dataset
        self.subgraph_type = subgraph_type
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataset)
        split_dataset = dataset.split('.')
        self.processed = os.path.join(root, processed_name)
        super(I2GNNDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.y_dim = self.data.y.size(-1)
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return self.root

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        dataset_base = self.dataset.split('.')[0]
        return [f'{dataset_base}_{self.subgraph_type}_{self.hops}hops.pt']

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        # assert np.min(begin) == 0
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        graphs, counts = load_graphs(self.raw_paths[0])
        data_list_all = [[self.adj2data(graph.adj().to_dense().numpy(), count.numpy()) for (graph, count) in zip(graphs, counts[self.subgraph_type])]]
        
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at'+save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    temp.append(self.pre_transform(data, self.hops))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)



class I2GNNDatasetRobustness(InMemoryDataset):
    # dataset class used fo the robustness experiments (does not read and write files but simply takes in input a list of nx graphs)
    def __init__(self, hops: int, graphs: List,
                 transform=None, pre_transform=None, pre_filter=None, url=None, root=None):
        if root is None:
            root = os.getcwd()
        self.hops = hops
        self.graphs = graphs
        super(I2GNNDatasetRobustness, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        self.data, self.slices = self.process_2()
        print(self.processed_dir)
        shutil.rmtree(self.processed_dir)
        # self.y_dim = self.data.y.size(-1)
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        # assert np.min(begin) == 0
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def process(self):
        pass

    def process_2(self):
        # process npy data into pyg.Data
        data_list = [self.adj2data(nx.to_numpy_array(graph), count.cpu().numpy()) for (graph, count) in self.graphs]
        # print('pre-transforming for data')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            temp = []
            for i, data in enumerate(data_list):
                # if i % 100 == 0:
                #     print('Pre-processing %d/%d' % (i, len(data_list)))
                temp.append(self.pre_transform(data, self.hops))
            data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        return data, slices




class I2GNNDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                return Batch.from_data_list(batch, follow_batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
                return type(elem)(*(collate(s) for s in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                return [collate(s) for s in zip(*batch)]

            raise TypeError('DataLoader found invalid type: {}'.format(
                type(elem)))

        super().__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda batch: collate(batch), **kwargs)
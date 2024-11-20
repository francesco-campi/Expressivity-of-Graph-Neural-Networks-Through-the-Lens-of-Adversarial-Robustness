from torch import nn, ones, empty, no_grad, long, tensor, Tensor
import networkx as nx
from torch_geometric.data import Data
from typing import Tuple, List
from copy import deepcopy
import multiprocessing
from time import time, sleep
from math import ceil
import random
import numpy as np
import pandas as pd
import torch

from .adversarial_attack_base import AdversarialAttackBase
from .greedy_attack import GreedyAttack, PreserveGreedyAttack
from ..dataset.counting_algorithm import subgraph_counting, subgraph_listing
from ..I2GNN.I2GNN_dataset import I2GNNDatasetRobustness, I2GNNDataLoader
from ..I2GNN.utils import create_subgraphs2

diameters = {
    "Triangle": 1,
    "2-Path": 2,
    "4-Clique": 1, 
    "Chordal cycle": 2,
    "Tailed triangle": 2,
    "3-Star": 2,
    "4-Cycle": 2,
    "3-Path": 3,
    "3-Star not ind.": 2,
}

def pre_transform(g, hops):
            return create_subgraphs2(g, hops)

class I2GNNBeamAttack(GreedyAttack):

    def __init__(self, n_samples: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", batch_size:int = 32, root = None):
        super().__init__(edge_deletion, edge_addition, device)
        self.n_samples = n_samples #n of adv graphs for which we compute the possible perturbations
        self.adversarial_graphs = None
        self.adversarial_counts = None
        self.batch_size =batch_size
        self.root = root

    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        best_adversarial_count = count # keep track of it in in the non preserving case
        if self.adversarial_graphs is None: #if previous computations have been done already keep alive all the possible paths
            self.adversarial_graphs = [graph]
        if self.adversarial_counts is None:
            self.adversarial_counts = [count]
        best_adversarial_graph = graph
        
        dataset = I2GNNDatasetRobustness(diameters[subgraph_type], [(graph, count)], pre_transform=pre_transform, root= self.root)
        dataloader = I2GNNDataLoader(dataset, shuffle=False)
        with no_grad():
            for data in dataloader: # only one instance
                data.to(self.device)
                best_adversarial_error = loss_fn(self.gnn(data), data.y).item()
        adversarial_error_history = [best_adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            perturbed_graphs = []
            for adversarial_graph, adversarial_count in zip(self.adversarial_graphs, self.adversarial_counts):
                perturbed_graphs.extend(self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type))
            error_perturbed_graphs = pd.Series(index=list(range(len(perturbed_graphs)))) # erros of the perturbed graphs

            dataset = I2GNNDatasetRobustness(diameters[subgraph_type], perturbed_graphs, pre_transform=pre_transform, root=self.root)
            dataloader = I2GNNDataLoader(dataset, shuffle=False, batch_size=self.batch_size)

            with no_grad():
                for j, data in enumerate(dataloader):
                    data.to(self.device)
                    preds = self.gnn(data)
                    for k, (pred, y) in enumerate(zip(preds, data.y)):
                        error_perturbed_graphs[j*self.batch_size + k] = self.loss_fn(pred, y).item()
            
            error_perturbed_graphs = error_perturbed_graphs.sort_values(ascending=False).drop_duplicates().head(self.n_samples) # select the best n_samples graphs 

            if len(error_perturbed_graphs) > 0 and error_perturbed_graphs.iloc[0] > best_adversarial_error: # better example has been found, update the adversarial features
                best_adversarial_error = error_perturbed_graphs.iloc[0]
                best_adversarial_graph = perturbed_graphs[error_perturbed_graphs.index[0]][0]
                best_adversarial_count = perturbed_graphs[error_perturbed_graphs.index[0]][1]
            # even if we haven't found any improvement we still try to see if different paths bring to better results
            adversarial_error_history.append(best_adversarial_error)
            # new graphs to perturb
            self.adversarial_graphs = [perturbed_graphs[i][0] for i in error_perturbed_graphs.index]
            self.adversarial_counts = [perturbed_graphs[i][1] for i in error_perturbed_graphs.index]
        
        return best_adversarial_graph, adversarial_error_history, best_adversarial_count


class I2GNNPreserveBeamAttack(PreserveGreedyAttack):

    def __init__(self, n_samples: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", batch_size:int = 32, root = None):
        super().__init__(edge_deletion, edge_addition, device)
        self.n_samples = n_samples #n of adv graphs for which we compute the possible perturbations
        self.adversarial_graphs = None
        self.batch_size = batch_size
        self.root = root
    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count # keep track of it in in the non preserving case
        if self.adversarial_graphs is None: #if previous computations have been done already keep alive all the possible paths
            self.adversarial_graphs = [graph]
        best_adversarial_graph = graph
        dataset = I2GNNDatasetRobustness(diameters[subgraph_type], [(graph, count)], pre_transform=pre_transform, root = self.root)
        dataloader = I2GNNDataLoader(dataset, shuffle=False)
        with no_grad():
            for data in dataloader: # only one instance
                data.to(self.device)
                best_adversarial_error = self.loss_fn(self.gnn(data), data.y).item()
        adversarial_error_history = [best_adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            perturbed_graphs = []
            for adversarial_graph in self.adversarial_graphs:
                perturbed_graphs.extend(self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type))
            if len(perturbed_graphs) == 0:
                break
            error_perturbed_graphs = pd.Series(index=list(range(len(perturbed_graphs)))) # erros of the perturbed graphs
            dataset = I2GNNDatasetRobustness(diameters[subgraph_type], perturbed_graphs, pre_transform=pre_transform, root=self.root)
            dataloader = I2GNNDataLoader(dataset, shuffle=False, batch_size=self.batch_size)
            
            with no_grad():
                for j, data in enumerate(dataloader):
                    data.to(self.device)
                    preds = self.gnn(data)
                    for k, (pred, y) in enumerate(zip(preds, data.y)):
                        error_perturbed_graphs[j*self.batch_size + k] = self.loss_fn(pred, y).item()
            error_perturbed_graphs = error_perturbed_graphs.sort_values(ascending=False).drop_duplicates().head(self.n_samples) # select the best n_samples graphs 

            if len(error_perturbed_graphs) > 0 and error_perturbed_graphs.iloc[0] > best_adversarial_error: # better example has been found, update the adversarial features
                best_adversarial_error = error_perturbed_graphs.iloc[0]
                best_adversarial_graph = perturbed_graphs[error_perturbed_graphs.index[0]][0]
            # even if we haven't found any improvement we still try to see if different paths bring to better results
            adversarial_error_history.append(best_adversarial_error)
            # new graphs to perturb
            self.adversarial_graphs = [perturbed_graphs[i][0] for i in error_perturbed_graphs.index]

        
        return best_adversarial_graph, adversarial_error_history, adversarial_count


class I2GNNSubstructurePreserveBeamAttack(I2GNNPreserveBeamAttack):

    def __init__(self, n_samples: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", batch_size:int = 32, root = None):
        super().__init__(n_samples, edge_deletion, edge_addition, device, batch_size, root)

    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs) -> Tuple[nx.Graph, list, Tensor]:
        subgraphs = subgraph_listing(graph, subgraph_type)
        self.keep_edges = []
        self.keep_non_edges = []
        for subgraph in subgraphs:
            nx_subgraph = graph.subgraph(subgraph)
            self.keep_edges.extend(nx.edges(nx_subgraph))
            self.keep_non_edges.extend(nx.non_edges(nx_subgraph))
        return super().find_adversarial_example(budget, gnn, loss_fn, graph, subgraph_type, count, **kwargs)
    

    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        edge_deleted_graphs = []
        edges = nx.edges(graph)
        for edge in edges:
            if edge in self.keep_edges:
                continue
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            if perturbed_graph.number_of_edges() == 0:
                continue
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_list = subgraph_listing(graph=subgraph, subgraph_type=subgraph_type)
            subgraph_count = tensor([len(subgraph_list)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_list = subgraph_listing(graph=perturbed_subgraph, subgraph_type=subgraph_type)
            perturbed_subgraph_count = tensor([len(perturbed_subgraph_list)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_subgraph_list != subgraph_list:
                continue
            perturbed_count = perturbed_count
            edge_deleted_graphs.append((perturbed_graph, perturbed_count))
        return edge_deleted_graphs


    def _edge_addition(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""
        edge_added_graphs = []
        non_edges = nx.non_edges(graph)
        for edge in non_edges:
            if edge in self.keep_non_edges:
                continue
            perturbed_graph = deepcopy(graph)
            perturbed_graph.add_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_list = subgraph_listing(graph=subgraph, subgraph_type=subgraph_type)
            subgraph_count = tensor([len(subgraph_list)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_list = subgraph_listing(graph=perturbed_subgraph, subgraph_type=subgraph_type)
            perturbed_subgraph_count = tensor([len(perturbed_subgraph_list)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_subgraph_list != subgraph_list:
                continue
            perturbed_count = perturbed_count
            edge_added_graphs.append((perturbed_graph, perturbed_count))
        return edge_added_graphs



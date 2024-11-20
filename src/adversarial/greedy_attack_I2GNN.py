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

from .greedy_attack import GreedyAttack, RandomGreedyAttack
from ..I2GNN.I2GNN_dataset import I2GNNDatasetRobustness, I2GNNDataLoader
from ..I2GNN.utils import create_subgraphs2
from ..dataset.counting_algorithm import subgraph_counting

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

class I2GNNGreedyAttack(GreedyAttack):

    def __init__(self, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu",  batch_size:int = 32, root = None):
        super().__init__(edge_deletion, edge_addition, device)
        self.batch_size = batch_size
        self.root = root

    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count
        adversarial_graph = graph

        dataset = I2GNNDatasetRobustness(diameters[subgraph_type], [(graph, count)], pre_transform=pre_transform, root= self.root)
        dataloader = I2GNNDataLoader(dataset, shuffle=False)
        with no_grad():
            for data in dataloader: # only one instance
                data.to(self.device)
                adversarial_error = loss_fn(self.gnn(data), data.y).item()
        adversarial_error_history = [adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            found = False
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            
            error_perturbed_graphs = pd.Series(index=list(range(len(perturbed_graphs)))) # erros of the perturbed graphs
            dataset = I2GNNDatasetRobustness(diameters[subgraph_type], perturbed_graphs, pre_transform=pre_transform, root=self.root)
            dataloader = I2GNNDataLoader(dataset, shuffle=False, batch_size=self.batch_size)

            with no_grad():
                for j, data in enumerate(dataloader):
                    data.to(self.device)
                    preds = self.gnn(data)
                    for k, (pred, y) in enumerate(zip(preds, data.y)):
                        error_perturbed_graphs[j*self.batch_size + k] = self.loss_fn(pred, y).item()
            
            id_max = error_perturbed_graphs.idxmax() # select the best n_samples graphs 
            error_perturbed_graph = error_perturbed_graphs[id_max]
            # if we find a more effective perturbation set it as the new best
            if error_perturbed_graph > adversarial_error:
                found = True
                adversarial_error = error_perturbed_graph
                adversarial_graph = perturbed_graphs[id_max][0]
                adversarial_count = perturbed_graphs[id_max][1]
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error)
        
        return adversarial_graph, adversarial_error_history, adversarial_count

    

class I2GNNRandomGreedyAttack(RandomGreedyAttack):

    def __init__(self, n_samples: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu",  batch_size:int = 32, root = None):
        super().__init__(n_samples, edge_deletion, edge_addition, device)
        self.batch_size = batch_size
        self.root = root

    
    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count
        adversarial_graph = graph

        dataset = I2GNNDatasetRobustness(diameters[subgraph_type], [(graph, count)], pre_transform=pre_transform, root= self.root)
        dataloader = I2GNNDataLoader(dataset, shuffle=False)
        with no_grad():
            for data in dataloader: # only one instance
                data.to(self.device)
                adversarial_error = loss_fn(self.gnn(data), data.y).item()
        adversarial_error_history = [adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            
            error_perturbed_graphs = pd.Series(index=list(range(len(perturbed_graphs)))) # erros of the perturbed graphs
            dataset = I2GNNDatasetRobustness(diameters[subgraph_type], perturbed_graphs, pre_transform=pre_transform, root=self.root)
            dataloader = I2GNNDataLoader(dataset, shuffle=False, batch_size=self.batch_size)

            with no_grad():
                for j, data in enumerate(dataloader):
                    data.to(self.device)
                    preds = self.gnn(data)
                    for k, (pred, y) in enumerate(zip(preds, data.y)):
                        error_perturbed_graphs[j*self.batch_size + k] = self.loss_fn(pred, y).item()
            
            id_max = error_perturbed_graphs.idxmax() # select the best n_samples graphs 
            error_perturbed_graph = error_perturbed_graphs[id_max]
            # if we find a more effective perturbation set it as the new best
            if error_perturbed_graph > adversarial_error:
                adversarial_error = error_perturbed_graph
                adversarial_graph = perturbed_graphs[id_max][0]
                adversarial_count = perturbed_graphs[id_max][1]

            adversarial_error_history.append(adversarial_error)
        
        return adversarial_graph, adversarial_error_history, adversarial_count
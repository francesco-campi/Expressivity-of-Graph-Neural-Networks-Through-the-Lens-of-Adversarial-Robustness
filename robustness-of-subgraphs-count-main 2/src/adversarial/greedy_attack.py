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

from .adversarial_attack_base import AdversarialAttackBase
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

class GreedyAttack(AdversarialAttackBase):

    def __init__(self, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu"):
        super().__init__()
        self.edge_deletion = edge_deletion
        self.edge_addition = edge_addition
        self.device = device

    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count.to(self.device)
        adversarial_graph = graph

        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), adversarial_count).item()
        adversarial_error_history = [adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            found = False
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            for perturbed_graph, perturbed_count in perturbed_graphs:
                with no_grad():
                    error_perturbed_graph = self.loss_fn(self.gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count).item()

                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    found = True
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
                    adversarial_count = perturbed_count
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error)
        
        return adversarial_graph, adversarial_error_history, adversarial_count


    def _generate_perturbations(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of perturbed graphs accordign to the given perturbation types adn their updated count.
        They are stored in a list of tuples, where eahc tuple contain the preturbed graph and its count"""

        perturbed_graphs = []
        if self.edge_deletion:
            perturbed_graphs.extend(self._edge_deletion(graph, count, subgraph_type))
        if self.edge_addition:
            perturbed_graphs.extend(self._edge_addition(graph, count, subgraph_type))
        # TODO: node perturbations
        return perturbed_graphs

    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, int]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        edge_deleted_graphs = []
        edges = nx.edges(graph)
        for edge in edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            if perturbed_graph.number_of_edges() == 0:
                continue
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count
            edge_deleted_graphs.append((perturbed_graph, perturbed_count))
        return edge_deleted_graphs


    def _edge_addition(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, int]]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""
        edge_added_graphs = []
        non_edges = nx.non_edges(graph)
        for edge in non_edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.add_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count
            edge_added_graphs.append((perturbed_graph, perturbed_count))
        return edge_added_graphs



    def _generate_egonet(self, graph: nx.Graph, root: int, depth: int)->List[int]:
        """Generates an egonet of given depth starting from the root node"""
        # to generate the egonet we look for neighbours iteratively
        graph_copy = deepcopy(graph)
        egonet = [root]
        neighbors = [root]
        visited = {root} # here we need to search for an element

        for i in range(depth):
            queue = neighbors
            neighbors = []
            for node in queue:
                for neighbor in graph_copy.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbors.append(neighbor)
            if neighbors == []:
                break
            egonet.extend(neighbors)
        return egonet
    
    def _generate_gnn_input(self, graph: nx.Graph)->Data:
        """Creates from a networkx graph a Data instance, which is the input a a pytorch geometric model."""
        num_edges = graph.number_of_edges()
        x = ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
        edge_index = empty(2, 2*num_edges, dtype=long)
        for i, edge in enumerate(graph.edges()):
            edge_index[0,i] = edge[0]
            edge_index[1,i] = edge[1]
            edge_index[0, i+num_edges] = edge[1]
            edge_index[1, i+num_edges] = edge[0]
        return Data(x=x, edge_index=edge_index).to(self.device)
    

class RandomGreedyAttack(GreedyAttack):

    def __init__(self, n_samples: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu"):
        super().__init__(edge_deletion, edge_addition, device)
        self.n_samples = n_samples
    
    def find_adversarial_example(self, budget: int, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.budget = budget
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count.to(self.device)
        adversarial_graph = graph

        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), adversarial_count).item()
        adversarial_error_history = [adversarial_error]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            for perturbed_graph, perturbed_count in perturbed_graphs:
                with no_grad():
                    error_perturbed_graph = self.loss_fn(self.gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count).item()

                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
                    adversarial_count = perturbed_count

            adversarial_error_history.append(adversarial_error)
        
        return adversarial_graph, adversarial_error_history, adversarial_count


    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        edge_deleted_graphs = []
        edges = np.array(nx.edges(graph))
        if len(edges) == 0:
            return edge_deleted_graphs
        # sampled_edges = random.sample(edges, self.n_samples)
        prob = self._generate_probabilities(graph, edges)
        sampled_edges_ids = np.random.choice(a=range(len(edges)), size= min(np.sum(prob > 0), self.n_samples), replace=False, p=prob)
        sampled_edges = [edges[i] for i in sampled_edges_ids]
        for edge in sampled_edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            if perturbed_graph.number_of_edges() == 0:
                print('no more edges')
                continue
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count.to(self.device)
            edge_deleted_graphs.append((perturbed_graph, perturbed_count))
        return edge_deleted_graphs


    def _edge_addition(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""
        edge_added_graphs = []
        non_edges = np.array(list(nx.non_edges(graph)))
        if len(non_edges) == 0:
            return edge_added_graphs
        # sampled_non_edges = random.sample(non_edges, self.n_samples)
        prob = self._generate_probabilities(graph, non_edges)
        sampled_non_edges_ids = np.random.choice(a=range(len(non_edges)), size= min(np.sum(prob > 0), self.n_samples), replace=False, p=prob)
        sampled_non_edges = [non_edges[i] for i in  sampled_non_edges_ids]
        for edge in sampled_non_edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.add_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count.to(self.device)
            edge_added_graphs.append((perturbed_graph, perturbed_count))
        return edge_added_graphs

    def _generate_probabilities(self, graph: nx.Graph, edges: np.ndarray) -> np.ndarray:
        #assigns to each edge a probability proportional to the degrees of its nodes
        prob = np.empty(edges.shape[0], dtype=float)
        for i, edge in enumerate(edges):
            prob[i] = (graph.degree[edge[0]] + graph.degree[edge[1]])**2
        # normalize the probability
        sum_prob = np.sum(prob)
        if sum_prob == 0.:
            prob = np.ones(len(edges)) / len(edges)
        else:
            prob = prob / sum_prob
        return prob


class PreserveGreedyAttack(GreedyAttack):

    def __init__(self, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu"):
        super().__init__(edge_deletion, edge_addition, device)

    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        edge_deleted_graphs = []
        edges = nx.edges(graph)
        for edge in edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            if perturbed_graph.number_of_edges() == 0:
                continue
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_count != count:
                continue
            perturbed_count = perturbed_count.to(self.device)
            edge_deleted_graphs.append((perturbed_graph, perturbed_count))
        return edge_deleted_graphs


    def _edge_addition(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""
        edge_added_graphs = []
        non_edges = nx.non_edges(graph)
        for edge in non_edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.add_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_count != count:
                continue
            perturbed_count = perturbed_count.to(self.device)
            edge_added_graphs.append((perturbed_graph, perturbed_count))
        return edge_added_graphs
    

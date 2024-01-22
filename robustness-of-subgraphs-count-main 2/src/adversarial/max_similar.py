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

from .beam_attack import SubstructurePreserveBeamAttack, PreserveBeamAttack
from ..dataset.counting_algorithm import subgraph_counting, subgraph_listing


diameters = {
    "Triangle": 2,
    "2-Path": 2,
    "4-Clique": 1, 
    "Chordal cycle": 2,
    "Tailed triangle": 2,
    "3-Star": 2,
    "4-Cycle": 2,
    "3-Path": 3,
    "3-Star not ind.": 2,
}


class MaximizeSimilar(PreserveBeamAttack):
    def __init__(self, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", preserve:bool = False):
        self.edge_addition = edge_addition
        self.edge_deletion = edge_deletion
        self.device = device
        self.preserve = preserve

    def find_adversarial_example(self, budget: int, graph: nx.Graph, subgraph_type: str, count: Tensor, similar_subgraph_type: str, increase: bool, **kwargs) -> Tuple[nx.Graph, list, Tensor]:
        if self.preserve:
            subgraphs = subgraph_listing(graph, subgraph_type)
            self.keep_edges = []
            self.keep_non_edges = []
            for subgraph in subgraphs:
                nx_subgraph = graph.subgraph(subgraph)
                self.keep_edges.extend(nx.edges(nx_subgraph))
                self.keep_non_edges.extend(nx.non_edges(nx_subgraph))
        
        self.similar_subgraph_type = similar_subgraph_type

        if increase:
            self.compare = lambda new, old: new > old
        else:
            self.compare = lambda new, old: new < old

        adversarial_count = count.to(self.device) # keep track of it in in the non preserving case
        best_adversarial_graph = graph
        self.best_similar_count = subgraph_counting(graph, similar_subgraph_type)

        for i in range(budget):

            perturbed_graph = self._new_perturbation(graph=best_adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            if perturbed_graph is None:
                return best_adversarial_graph, best_similar_count

            best_adversarial_graph = perturbed_graph[0]
            best_similar_count = perturbed_graph[1]

        return best_adversarial_graph, best_similar_count


    def _new_perturbation(self, graph, count, subgraph_type):
        if self.edge_deletion:
            if self.preserve:
                pert_graph_d = self._edge_deletion_preserve(graph, count, subgraph_type)
            else:
                pert_graph_d = self._edge_deletion(graph, count, subgraph_type)
            if pert_graph_d is not None:
                graph = pert_graph_d[0]
                self.best_similar_count = pert_graph_d[1]
        if self.edge_addition:
            if self.preserve:
                pert_graph_a = self._edge_addition_preserve(graph, count, subgraph_type)
            else:
                pert_graph_a = self._edge_addition(graph, count, subgraph_type)
            if pert_graph_a is not None:
                return pert_graph_a
        if pert_graph_a is None and pert_graph_d is None:
            return None
        else: # only deletion worked
            return pert_graph_d

    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->Tuple[nx.Graph, Tensor]:
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
            perturbed_similar_count = subgraph_counting(perturbed_graph, self.similar_subgraph_type)
            if self.compare(perturbed_similar_count, self.best_similar_count):
                return (perturbed_graph, perturbed_similar_count)
        return None

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
            perturbed_similar_count = subgraph_counting(perturbed_graph, self.similar_subgraph_type)
            if self.compare(perturbed_similar_count, self.best_similar_count):
                return (perturbed_graph, perturbed_similar_count)
        return None


    def _edge_deletion_preserve(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
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
            subgraph_count = tensor([len(subgraph_list)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_list = subgraph_listing(graph=perturbed_subgraph, subgraph_type=subgraph_type)
            perturbed_subgraph_count = tensor([len(perturbed_subgraph_list)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_subgraph_list != subgraph_list:
                continue
            perturbed_count = perturbed_count.to(self.device)
            perturbed_similar_count = subgraph_counting(perturbed_graph, self.similar_subgraph_type)
            if self.compare(perturbed_similar_count, self.best_similar_count):
                return (perturbed_graph, perturbed_similar_count)
        return None


    def _edge_addition_preserve(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor]]:
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
            subgraph_count = tensor([len(subgraph_list)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_list = subgraph_listing(graph=perturbed_subgraph, subgraph_type=subgraph_type)
            perturbed_subgraph_count = tensor([len(perturbed_subgraph_list)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            if perturbed_subgraph_list != subgraph_list:
                continue
            perturbed_count = perturbed_count.to(self.device)
            perturbed_similar_count = subgraph_counting(perturbed_graph, self.similar_subgraph_type)
            if self.compare(perturbed_similar_count, self.best_similar_count):
                return (perturbed_graph, perturbed_similar_count)
        return None
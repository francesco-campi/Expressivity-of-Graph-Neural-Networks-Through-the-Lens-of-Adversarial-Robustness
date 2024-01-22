import random
import copy

import networkx as nx

class GraphPermutation:
    """Class that generates permutation on the graphs.
    """
    def __init__(self) -> None:
        #add teh possoibility to have a seed
        pass

    def edge_deletion(self, graph: nx.Graph, num: int = 1,seed: int = None) -> nx.Graph:
        """Deletion of an edge choosen uniformly.

        :param graph: graph to permute
        :type graph: nx.Graph
        """
        if seed is not None:
            random.seed(seed)
        perm_graph = copy.deepcopy(graph) # graph is passed by reference, but we don't want to modify the original
        # choose uniformly the edge to delete
        for i in range(num):
            edges = list(nx.edges(perm_graph))
            edge_to_delete = random.randint(0, len(edges)-1)
            perm_graph.remove_edge(*edges[edge_to_delete])
        return perm_graph

    def edge_creation(self, graph: nx.Graph, num: int = 1, seed: int = None) -> nx.Graph:
        """Addition of a non-exisiting edge choosen uniformly.

        :param graph: graph to permute
        :type graph: nx.Graph
        """
        if seed is not None:
            random.seed(seed)
        perm_graph = copy.deepcopy(graph)
        # choose uniformly an edge to add
        for i in range(num):
            non_edges = list(nx.non_edges(perm_graph))
            edge_to_add = random.randint(0, len(non_edges)-1)
            perm_graph.add_edge(*non_edges[edge_to_add])
        return perm_graph

    def edge_replacement(self, graph: nx.Graph, num : int = 1, seed: int = None) -> nx.Graph:
        """Deletion of an edge choosen uniformly and addition of a non-exisiting edge choosen also uniformly.

        :param graph: graph to permute
        :type graph: nx.Graph
        """
        # firstly delete the edge
        graph = self.edge_deletion(graph, num,  seed)
        #then add an edge
        graph = self.edge_creation(graph, num, seed)
        return graph
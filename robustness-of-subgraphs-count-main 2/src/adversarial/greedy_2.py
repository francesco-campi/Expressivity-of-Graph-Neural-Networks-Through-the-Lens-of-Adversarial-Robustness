from torch import nn, ones, empty, no_grad, long, tensor, Tensor
import networkx as nx
from torch_geometric.data import Data
from typing import Tuple, List
from copy import deepcopy
from  multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory
from time import time, sleep
from math import ceil
from sys import getsizeof
import numpy as np

from .adversarial_attack_base import AdversarialAttackBase
from ..dataset.counting_algorithm import subgraph_counting

diameters = {
    "g31": 2,
    "g32": 2,
    "g41": 1, 
    "g42": 2,
    "g43": 2,
    "g44": 2,
    "g45": 2,
    "g46": 3,
    "s3": 2,
}


class GreedyAttack0(AdversarialAttackBase):

    def __init__(self, budget: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", n_jobs: int = -1):
        super().__init__()
        self.budget = budget
        self.edge_deletion = edge_deletion
        self.edge_addition = edge_addition
        self.device = device
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def find_adversarial_example(self, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs)-> Tuple[nx.Graph, list, Tensor]:
        """Generates the best adversarial example found with a greeedy algorithm"""

        gnn.eval()
        self.gnn = gnn.to(self.device)
        self.loss_fn = loss_fn
        adversarial_count = count.to(self.device)
        adversarial_graph = graph
        adversarial_degrees = (0, 0)

        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), count)
        adversarial_error_history = [adversarial_error.item()]

        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            print(f"Budget {i}")
            found = False
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            max, min, avg = self._degree_statistics(graph)
            for perturbed_graph, perturbed_count, deg1, deg2 in perturbed_graphs:
                with no_grad():
                    error_perturbed_graph = self.loss_fn(self.gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count)
                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    found = True
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
                    adversarial_count = perturbed_count
                    adversarial_degrees = (deg1, deg2)
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error.item())
            print(f'Adversarial degrees: {adversarial_degrees}\nStatistics: max {max}, min {min}, avg {avg}')
        
        return adversarial_graph, adversarial_error_history, adversarial_count


    def _generate_perturbations(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor, int, int]]:
        """Returns a list of perturbed graphs accordign to the given perturbation types adn their updated count.
        They are stored in a list of tuples, where eahc tuple contain the preturbed graph and its count"""

        perturbed_graphs = []
        if self.edge_deletion:
            perturbed_graphs.extend(self._edge_deletion(graph, count, subgraph_type))
        if self.edge_addition:
            perturbed_graphs.extend(self._edge_addition(graph, count, subgraph_type))
        # TODO: node perturbations
        return perturbed_graphs

    def _edge_deletion(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor, int, int]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        edge_deleted_graphs = []
        edges = nx.edges(graph)
        for edge in edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)]).to(self.device)
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count.to(self.device)
            edge_deleted_graphs.append((perturbed_graph, perturbed_count, graph.degree[edge[0]], graph.degree[edge[1]]))
        return edge_deleted_graphs


    def _edge_addition(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->List[Tuple[nx.Graph, Tensor, int, int]]:
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
            perturbed_count = perturbed_count.to(self.device)
            edge_added_graphs.append((perturbed_graph, perturbed_count,  graph.degree[edge[0]], graph.degree[edge[1]]))
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
        x = ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
        edge_index = empty(2, graph.number_of_edges(), dtype=long)
        for i, edge in enumerate(graph.edges()):
            edge_index[0,i] = edge[0]
            edge_index[1,i] = edge[1]
        return Data(x=x, edge_index=edge_index).to(self.device)
    
    def _degree_statistics(self, graph: nx.Graph) -> Tuple[int, int, float]:
        min = graph.number_of_nodes()
        max = 0
        avg = 0.
        for _ , deg in graph.degree:
            avg += deg
            if deg > max:
                max = deg 
            if deg < min:
                min = deg
        avg = avg/graph.number_of_nodes()
        return max, min, avg

    

class GreedyAttack1(AdversarialAttackBase):

    


    def __init__(self, budget: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", n_jobs: int = -1):
        super().__init__()
        self.budget = budget
        self.edge_deletion = edge_deletion
        self.edge_addition = edge_addition
        self.device = device
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        self.gen_time = 0.
        self.eval_time = 0.

    def find_adversarial_example(self, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs):
        gnn.eval()
        adversarial_count = count.to(self.device)
        adversarial_graph = graph
        start = time()
        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), count)
        self.eval_time += time() - start
        adversarial_error_history = [adversarial_error.item()]
        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            found = False
            start = time()
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            self.gen_time += time() - start
            for perturbed_graph, perturbed_count in perturbed_graphs:
                start = time()
                with no_grad():
                    error_perturbed_graph = loss_fn(gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count)
                self.eval_time += time() - start
                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    found = True
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
                    adversarial_count = perturbed_count
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error.item())
        
        return adversarial_graph, adversarial_error_history


    def _generate_perturbations(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->list[Tuple[nx.Graph, Tensor]]:
        """Returns a list of perturbed graphs accordign to the given perturbation types adn their updated count.
        They are stored in a list of tuples, where eahc tuple contain the preturbed graph and its count"""

        adj = nx.to_numpy_array(graph)
        shm = SharedMemory(create=True, size=getsizeof(adj))
        shr_adj = np.ndarray(adj.shape, dtype=adj.dtype, buffer=shm.buf)
        np.copyto(shr_adj, adj)

        
        with Pool(self.n_jobs) as pool:
            processes = []
            if self.edge_deletion:
                edges = list(nx.edges(graph))
                processes.extend([pool.apply_async(self._edge_deletion, args=(shm.name, shr_adj.dtype, shr_adj.shape, edge, count, subgraph_type,)) for edge in edges])

            if self.edge_addition:
                non_edges = list(nx.non_edges(graph))
                processes.extend([pool.apply_async(self._edge_addition, args=(shm.name, shr_adj.dtype, shr_adj.shape, edge, count, subgraph_type,)) for edge in non_edges])
            # TODO: node perturbations

            perturbed_graphs = [p.get() for p in processes]
        return perturbed_graphs

    def _edge_deletion(self, shr_name, type, shape, edge: Tuple, count: Tensor, subgraph_type: str)->Tuple[nx.Graph, Tensor]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""
        
        shm = SharedMemory(shr_name)
        adj = np.ndarray(shape=shape, dtype=type, buffer=shm.buf)
        graph = nx.Graph(adj)
        perturbed_graph = deepcopy(graph)
        perturbed_graph.remove_edge(*edge)
        # recompute the substructure coutns of the perturbed graph
        egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
        subgraph = graph.subgraph(egonet)
        subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
        perturbed_subgraph = perturbed_graph.subgraph(egonet)
        perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
        perturbed_count = count + perturbed_subgraph_count - subgraph_count
        perturbed_count = perturbed_count.to(self.device)
        return (perturbed_graph, perturbed_count)


    def _edge_addition(self, shr_name, type, shape, edge: Tuple, count: Tensor, subgraph_type: str)->Tuple[nx.Graph, Tensor]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""

        shm = SharedMemory(shr_name)
        adj = np.ndarray(shape=shape, dtype=type, buffer=shm.buf)
        graph = nx.Graph(adj)
        perturbed_graph = deepcopy(graph)
        perturbed_graph.add_edge(*edge)
        # recompute the substructure coutns of the perturbed graph
        egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
        subgraph = graph.subgraph(egonet)
        subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
        perturbed_subgraph = perturbed_graph.subgraph(egonet)
        perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
        perturbed_count = count + perturbed_subgraph_count - subgraph_count
        perturbed_count = perturbed_count.to(self.device)
        return (perturbed_graph, perturbed_count)


    def _generate_egonet(self, graph: nx.Graph, root: int, depth: int)->list[int]:
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
        x = ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
        edge_index = empty(2, graph.number_of_edges(), dtype=long)
        for i, edge in enumerate(graph.edges()):
            edge_index[0,i] = edge[0]
            edge_index[1,i] = edge[1]
        return Data(x=x, edge_index=edge_index).to(self.device)

class GreedyAttack2(AdversarialAttackBase):

    


    def __init__(self, budget: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", n_jobs: int = -1):
        super().__init__()
        self.budget = budget
        self.edge_deletion = edge_deletion
        self.edge_addition = edge_addition
        self.device = device
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.gen_time = 0.
        self.eval_time = 0.

    def find_adversarial_example(self, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs):
        gnn.eval()
        adversarial_count = count.to(self.device)
        adversarial_graph = graph
        start = time()
        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), count)
        self.eval_time += time() - start
        adversarial_error_history = [adversarial_error.item()]
        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            found = False
            start = time()
            perturbed_graphs = self._generate_perturbations(graph=adversarial_graph, count=adversarial_count, subgraph_type=subgraph_type)
            self.gen_time += time() - start
            for perturbed_graph, perturbed_count in perturbed_graphs:
                start = time()
                with no_grad():
                    error_perturbed_graph = loss_fn(gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count)
                self.eval_time += time() - start
                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    found = True
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
                    adversarial_count = perturbed_count
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error.item())
        
        return adversarial_graph, adversarial_error_history


    def _generate_perturbations(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->list[Tuple[nx.Graph, Tensor]]:
        """Returns a list of perturbed graphs accordign to the given perturbation types adn their updated count.
        They are stored in a list of tuples, where eahc tuple contain the preturbed graph and its count"""
        
        perturbed_graphs = []
        with multiprocessing.Pool(self.n_jobs) as pool:
            processes = []
            if self.edge_deletion:
                edges = list(nx.edges(graph))
                # create batches of edges
                dim_batches = ceil(len(edges)/self.n_jobs)
                edges_batches = [edges[i*dim_batches:min(len(edges), (i + 1)*dim_batches)] for i in range(self.n_jobs)]
                processes.extend([pool.apply_async(self._edge_deletion, args=(graph, edges, count, subgraph_type,)) for edges in edges_batches])

            if self.edge_addition:
                non_edges = list(nx.non_edges(graph))
                # create batches of edges
                dim_batches = ceil(len(non_edges)/self.n_jobs)
                non_edges_batches = [non_edges[i*dim_batches:min(len(non_edges), (i + 1)*dim_batches)] for i in range(self.n_jobs)]
                processes.extend([pool.apply_async(self._edge_addition, args=(graph, edges, count, subgraph_type,)) for edges in non_edges_batches])
            for p in processes:
                perturbed_graphs.extend(p.get())
            # TODO: node perturbations
            # print("Start processes....")
            # start = time()
            # processes = [pool.apply_async(task, ()) for i in range(245)]
            # print(f"processes creation took: {time() - start}")
            # start = time()
            # result = [p.get() for p in processes]
            # print(f"Processes finished in {time() - start}")
        return perturbed_graphs

    def _edge_deletion(self, graph: nx.Graph, edges: list[Tuple], count: Tensor, subgraph_type: str)->list[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""
        
        perturbed = []
        for edge in edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.remove_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count.to(self.device)
            perturbed.append((perturbed_graph, perturbed_count))
        return perturbed


    def _edge_addition(self, graph: nx.Graph, edges: list[Tuple], count: Tensor, subgraph_type: str)->list[Tuple[nx.Graph, Tensor]]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""

        perturbed = []
        for edge in edges:
            perturbed_graph = deepcopy(graph)
            perturbed_graph.add_edge(*edge)
            # recompute the substructure coutns of the perturbed graph
            egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
            subgraph = graph.subgraph(egonet)
            subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
            perturbed_subgraph = perturbed_graph.subgraph(egonet)
            perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
            perturbed_count = count + perturbed_subgraph_count - subgraph_count
            perturbed_count = perturbed_count.to(self.device)
            perturbed.append((perturbed_graph, perturbed_count))
        return perturbed


    def _generate_egonet(self, graph: nx.Graph, root: int, depth: int)->list[int]:
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
        x = ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
        edge_index = empty(2, graph.number_of_edges(), dtype=long)
        for i, edge in enumerate(graph.edges()):
            edge_index[0,i] = edge[0]
            edge_index[1,i] = edge[1]
        return Data(x=x, edge_index=edge_index).to(self.device)

class GreedyAttack3(AdversarialAttackBase):

    def __init__(self, budget: int, edge_deletion: bool = True, edge_addition: bool = True, device: str = "cpu", n_jobs: int = -1):
        super().__init__()
        self.budget = budget
        self.edge_deletion = edge_deletion
        self.edge_addition = edge_addition
        self.device = device
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.gen_time = 0.
        self.eval_time = 0.

    def find_adversarial_example(self, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs):
        gnn.eval()
        self.gnn = gnn
        self.loss_fn = loss_fn
        count = count.to(self.device)
        adversarial_graph = graph
        start = time()
        with no_grad():
            adversarial_error = loss_fn(gnn(self._generate_gnn_input(graph)).flatten(), count)
        self.eval_time += time() - start
        adversarial_error_history = [adversarial_error]
        for i in range(self.budget): # might make sense to consider a distance function between graphs... (to ensure no to go out of distribtion)
            found = False
            start = time()
            perturbed_graphs = self._generate_perturbations(graph=graph, count=count, subgraph_type=subgraph_type)
            self.gen_time += time() - start
            for perturbed_graph, error_perturbed_graph in perturbed_graphs:
                # if we find a more effective perturbation set it as the new best
                if error_perturbed_graph > adversarial_error:
                    found = True
                    adversarial_error = error_perturbed_graph
                    adversarial_graph = perturbed_graph
            if found == False:
                # no improovement have been found
                break
            adversarial_error_history.append(adversarial_error)
        
        self.gen_time -= self.eval_time
        return adversarial_graph, adversarial_error_history


    def _generate_perturbations(self, graph: nx.Graph, count: Tensor, subgraph_type: str)->list[Tuple[nx.Graph, Tensor]]:
        """Returns a list of perturbed graphs accordign to the given perturbation types adn their updated count.
        They are stored in a list of tuples, where eahc tuple contain the preturbed graph and its count"""

        pool = multiprocessing.Pool(self.n_jobs)
        processes = []
        if self.edge_deletion:
            edges = nx.edges(graph)
            processes.extend([pool.apply_async(self._edge_deletion, args=(graph, edge, count, subgraph_type,)) for edge in edges])
        if self.edge_addition:
            non_edges = nx.non_edges(graph)
            processes.extend([pool.apply_async(self._edge_addition, args=(graph, edge, count, subgraph_type,)) for edge in non_edges])
            
        # TODO: node perturbations
        perturbed_graphs = [p.get() for p in processes]
        return perturbed_graphs

    def _edge_deletion(self, graph: nx.Graph, edge: Tuple, count: Tensor, subgraph_type: str)->Tuple[nx.Graph, Tensor]:
        """Returns a list of all the possible perturbed graphs obtained by deleting one edge with the updated substructure counting.
        The substructure count for the perturbed graph is updated taking the egonet of 
        depth equal to the diameter of the substructure. In this case and edge is perturbed and one of the two end nodes of the 
        edge is used to generate the egonet, and the graph used is the  original because it generated the biggest egonet."""

        perturbed_graph = deepcopy(graph)
        perturbed_graph.remove_edge(*edge)
        # recompute the substructure coutns of the perturbed graph
        egonet = self._generate_egonet(graph, edge[0], diameters[subgraph_type])
        subgraph = graph.subgraph(egonet)
        subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
        perturbed_subgraph = perturbed_graph.subgraph(egonet)
        perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
        perturbed_count = count + perturbed_subgraph_count - subgraph_count
        perturbed_count = perturbed_count.to(self.device)
        start = time()
        with no_grad():
            error_perturbed_graph = self.loss_fn(self.gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count)
        self.eval_time += time() - start
        return (perturbed_graph, error_perturbed_graph)


    def _edge_addition(self, graph: nx.Graph, edge: Tuple, count: Tensor, subgraph_type: str)->Tuple[nx.Graph, Tensor]:
        """Returns a list of all the possible perturbed graphs obtained by adding one edge"""
        perturbed_graph = deepcopy(graph)
        perturbed_graph.add_edge(*edge)
        # recompute the substructure coutns of the perturbed graph
        egonet = self._generate_egonet(perturbed_graph, edge[0], diameters[subgraph_type])
        subgraph = graph.subgraph(egonet)
        subgraph_count = tensor([subgraph_counting(graph=subgraph, subgraph_type=subgraph_type)])
        perturbed_subgraph = perturbed_graph.subgraph(egonet)
        perturbed_subgraph_count = tensor([subgraph_counting(graph=perturbed_subgraph, subgraph_type=subgraph_type)])
        perturbed_count = count + perturbed_subgraph_count - subgraph_count
        perturbed_count = perturbed_count.to(self.device)
        start = time()
        with no_grad():
            error_perturbed_graph = self.loss_fn(self.gnn(self._generate_gnn_input(perturbed_graph)).flatten(), perturbed_count)
        self.eval_time += time() - start
        return (perturbed_graph, error_perturbed_graph)


    def _generate_egonet(self, graph: nx.Graph, root: int, depth: int)->list[int]:
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
        x = ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
        edge_index = empty(2, graph.number_of_edges(), dtype=long)
        for i, edge in enumerate(graph.edges()):
            edge_index[0,i] = edge[0]
            edge_index[1,i] = edge[1]
        return Data(x=x, edge_index=edge_index).to(self.device)

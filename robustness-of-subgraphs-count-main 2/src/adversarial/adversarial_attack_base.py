from abc import ABC, abstractmethod
from torch import nn, Tensor
import networkx as nx

class AdversarialAttackBase(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def find_adversarial_example(self, gnn: nn.Module, loss_fn: nn.Module, graph: nx.Graph, subgraph_type: str, count: Tensor, **kwargs):
        """Method that returns the most effective adversarial example"""
        # apply the optimization strategy to find the best adversarial example. and the relative error
        error = 0.
        return graph, error
import os
import sys
# add to the path the source files
sys.path.append(os.getcwd())

import pytest
from dgl.data.utils import load_graphs
import dgl
import networkx as nx
from matplotlib import pyplot as plt

from src.dataset.counting_algorithm import subgraph_counting, subgraph_counting_all
from src.adversarial.greedy_attack import GreedyAttack

@pytest.fixture
def data():
    graphs_1, counts_1 = load_graphs("tests/data/robustness_10_er_10.bin")
    return[graphs_1, counts_1]

def test_update_edge_deletion(data):
    ga = GreedyAttack(1)
    graphs = data[0]
    counts = data[1]
    for i, graph in enumerate(graphs):
        graph = nx.Graph(dgl.to_networkx(graph))
        for subgraph in counts.keys():
            for pert_graph, pert_count in ga._edge_deletion(graph, counts[subgraph][i], subgraph):
                real_pert_count = subgraph_counting(pert_graph, subgraph)
                assert real_pert_count == pert_count, f'real count: {real_pert_count}, computed_count: {pert_count}'
                for pert_graph2, pert_count2 in ga._edge_deletion(pert_graph, pert_count, subgraph):
                    real_pert_count2 = subgraph_counting(pert_graph2, subgraph)
                    if real_pert_count2 != pert_count2:
                        nx.draw(pert_graph, with_labels=True)
                        plt.show()
                        nx.draw(pert_graph2, with_labels=True)
                        plt.show()
                    assert real_pert_count2 == pert_count2, f'real count: {real_pert_count2}, computed_count: {pert_count2}'

def test_update_edge_addition(data):
    ga = GreedyAttack(1)
    graphs = data[0]
    counts = data[1]
    for i, graph in enumerate(graphs):
        graph = nx.Graph(dgl.to_networkx(graph))
        for subgraph in counts.keys():
            for pert_graph, pert_count in ga._edge_addition(graph, counts[subgraph][i], subgraph):
                real_pert_count = subgraph_counting(pert_graph, subgraph)
                assert real_pert_count == pert_count, f'real count: {real_pert_count}, computed_count: {pert_count}'
                for pert_graph2, pert_count2 in ga._edge_addition(pert_graph, pert_count, subgraph):
                    real_pert_count2 = subgraph_counting(pert_graph2, subgraph)
                    if real_pert_count2 != pert_count2:
                        nx.draw(pert_graph, with_labels=True)
                        plt.show()
                        nx.draw(pert_graph2, with_labels=True)
                        plt.show()
                    assert real_pert_count2 == pert_count2, f'real count: {real_pert_count2}, computed_count: {pert_count2}'

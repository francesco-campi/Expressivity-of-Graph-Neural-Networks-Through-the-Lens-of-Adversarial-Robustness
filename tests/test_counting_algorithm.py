# imports
import os
import sys
# add to the path the source files
sys.path.append(os.getcwd())

import pytest
from dgl.data.utils import load_graphs
import dgl
import networkx as nx
import numpy as np

from src.dataset.counting_algorithm import subgraph_counting, subgraph_counting_all, subgraph_listing, subgraph_listing_all

@pytest.fixture
def data_1():
    graphs_1, counts_1 = load_graphs("tests/data/dataset1.bin")
    return[graphs_1, counts_1]

@pytest.fixture
def data_2():
    graphs_2, counts_2 = load_graphs("tests/data/dataset2.bin")
    return [graphs_2, counts_2]

def test_count_star(data_1, data_2):
    graphs_1 = data_1[0]
    counts_1 = data_1[1]['star']
    graphs_2 = data_2[0]
    counts_2 = data_2[1]['star']

    for graph, count in zip(graphs_1, counts_1):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, '3-Star not ind.')
        assert count == new_count, "Stars count is wrong!"

    for graph, count in zip(graphs_2, counts_2):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, '3-Star not ind.')
        assert count == new_count, "Stars count is wrong!"


def test_count_triangle(data_1, data_2):
    graphs_1 = data_1[0]
    counts_1 = data_1[1]['triangle']
    graphs_2 = data_2[0]
    counts_2 = data_2[1]['triangle']

    for graph, count in zip(graphs_1, counts_1):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Triangle')
        assert count == new_count, "Triangles count is wrong!"

    for graph, count in zip(graphs_2, counts_2):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Triangle')
        assert count == new_count, "Triangles count is wrong!"

def test_count_tailed_triangle(data_1, data_2):
    graphs_1 = data_1[0]
    counts_1 = data_1[1]['tailed_triangle']
    graphs_2 = data_2[0]
    counts_2 = data_2[1]['tailed_triangle']

    for graph, count in zip(graphs_1, counts_1):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Tailed triangle')
        assert count == new_count, "Tailed triangles count is wrong!"

    for graph, count in zip(graphs_2, counts_2):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Tailed triangle')
        assert count == new_count, "Tailed triangles count is wrong!"

def test_count_chordal_cycle(data_1, data_2):
    graphs_1 = data_1[0]
    counts_1 = data_1[1]['chordal_cycle']
    graphs_2 = data_2[0]
    counts_2 = data_2[1]['chordal_cycle']

    for graph, count in zip(graphs_1, counts_1):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Chordal cycle')
        assert count == new_count, "Chordal cycles count is wrong!"

    for graph, count in zip(graphs_2, counts_2):
        graph = dgl.to_networkx(graph).to_undirected()
        new_count = subgraph_counting(graph, 'Chordal cycle')
        assert count == new_count, "Chordal cylces count is wrong!"


def test_consistence_countig_function(data_1, data_2):
    graphs_1 = data_1[0]
    graphs_2 = data_2[0]
    
    for graph in graphs_1:
        graph = dgl.to_networkx(graph).to_undirected()
        all_counts = subgraph_counting_all(graph)
        keys = list(all_counts.keys())
        for key in keys:
            single_count = subgraph_counting(graph, key)
            assert single_count == all_counts[key], f"Count of {key} are not consistent betwee single counting function and global counting"

    for graph in graphs_2:
        graph = dgl.to_networkx(graph).to_undirected()
        all_counts = subgraph_counting_all(graph)
        keys = list(all_counts.keys())
        for key in keys:
            single_count = subgraph_counting(graph, key)
            assert single_count == all_counts[key], f"Count of {key} are not consistent between single counting function and global counting"

######## TEST LISTING ALGORITHM ########

def test_consistency_listing(data_1, data_2):
    graphs_1 = data_1[0]
    graphs_2 = data_2[0]
    
    for graph in graphs_1:
        graph = dgl.to_networkx(graph).to_undirected()
        all_subgraphs = subgraph_listing_all(graph)
        keys = list(all_subgraphs.keys())
        for key in keys:
            single_subgraphs = subgraph_listing(graph, key)
            assert single_subgraphs == all_subgraphs[key], f"Count of {key} are not consistent betwee single function and global"

    for graph in graphs_2:
        graph = dgl.to_networkx(graph).to_undirected()
        all_subgraphs = subgraph_listing_all(graph)
        keys = list(all_subgraphs.keys())
        for key in keys:
            single_subgraphs = subgraph_listing(graph, key)
            assert single_subgraphs == all_subgraphs[key], f"Count of {key} are not consistent betwee single function and global"

def test_listing(data_1, data_2):
    graphs_1 = data_1[0]
    graphs_2 = data_2[0]
    
    for graph in graphs_1:
        graph = dgl.to_networkx(graph).to_undirected()
        subgraphs = subgraph_listing_all(graph)
        counts = subgraph_counting_all(graph)
        keys = list(subgraphs.keys())
        for key in keys:
            assert len(subgraphs[key]) == counts[key], f"Count and listing of {key} are not consistent"

    for graph in graphs_2:
        graph = dgl.to_networkx(graph).to_undirected()
        subgraphs = subgraph_listing_all(graph)
        counts = subgraph_counting_all(graph)
        keys = list(subgraphs.keys())
        for key in keys:
            assert len(subgraphs[key]) == counts[key], f"Count and listing of {key} are not consistent"



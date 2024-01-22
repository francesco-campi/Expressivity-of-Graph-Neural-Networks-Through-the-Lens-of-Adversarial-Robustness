import numpy as np
import os
import sys
from dgl import to_networkx, load_graphs
import torch
import networkx as nx
from statistics import mean
from typing import List
import copy
import json


sys.path.append(os.getcwd())
from src.dataset.counting_algorithm import subgraph_counting
from src.adversarial.greedy_attack import GreedyAttack, RandomGreedyAttack, PreserveGreedyAttack
from src.adversarial.beam_attack import  PreserveBeamAttack, SubstructurePreserveBeamAttack, BeamAttack
from src.adversarial.max_similar import MaximizeSimilar
from src.metrics.L1_based import L1LossCount, L1LossStd
from src.baseline.model_gcn import GIN
from src.ppgn.ppgn import PPGN

import logging
from sacred import Experiment
import seml
import torch_geometric.data



ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


def generate_gnn_input(graph: nx.Graph, device)->torch_geometric.data.Data:
    """Creates from a networkx graph a Data instance, which is the input a a pytorch geometric model."""
    num_edges = graph.number_of_edges()
    x = torch.ones(graph.number_of_nodes(), 1) # no improovement by using more channels in the first layer
    edge_index = torch.empty(2, 2*num_edges, dtype=torch.long)
    for i, edge in enumerate(graph.edges()):
        edge_index[0,i] = edge[0]
        edge_index[1,i] = edge[1]
        edge_index[0, i+num_edges] = edge[1]
        edge_index[1, i+num_edges] = edge[0]
    return torch_geometric.data.Data(x=x, edge_index=edge_index).to(device)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(dataset_path: str, graph_id: int, models_path: str, model_architecture: str, subgraphs: List[str], seed: int, budgets: List[int], edge_addition: bool,
        edge_deletion: bool, adversarial_graphs_folder: str, device: str, increase:bool, preserve: bool):
    """The program gets in input one test graph to analyze the robustess and a set of models. For each model the adversarial example obtained from 
    the imput graph is computed in parallel.
    """
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    subgraph = subgraphs[0]
    similar_subgraph = subgraphs[1]
    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset_path}, models archotecture: {model_architecture}, subgraph: {subgraph}, seed: {seed}')
    
    attack = MaximizeSimilar(edge_addition=edge_addition, edge_deletion=edge_deletion, device=device, preserve=preserve)

    os.makedirs(adversarial_graphs_folder, exist_ok=True)

    # load the graph (graph_id-th one that is correctly predicted)
    graphs, counts = load_graphs(dataset_path)
    with open(f"{dataset_path.split('.')[0]}_{model_architecture}.json", 'r') as f:
        adversarial_ids = json.load(f)
    graph_idx = adversarial_ids[subgraph][graph_id]
       
    graph = nx.Graph(to_networkx(graphs[graph_idx]))
    count = counts[subgraph][graph_idx]
    count_std = counts[subgraph].std()

    # load the model
    model_dict = f"{models_path}/{model_architecture}_{subgraph}_{seed}.pth"
    model_params = f"{models_path}/{model_architecture}_{subgraph}_{seed}.json"
    with open(model_params, 'r') as fp:
        h_params = json.load(fp)
    if model_architecture == 'GIN':
        gnn = GIN(**h_params)
    elif model_architecture == 'PPGN':
        gnn = PPGN(**h_params)
    else:
        raise ValueError("The architecture is not supported!")
    gnn.load_state_dict(torch.load(model_dict, map_location=torch.device(device)))
    gnn.eval()


    # find adversarial example
    adversarial_errors = []
    adversarial_prediction = []
    adversarial_counts = []
    adversarial_graph = graph
    adversarial_count = count

    with torch.no_grad():
        test_prediction = gnn(generate_gnn_input(graph, device)).item()
        test_error = (gnn(generate_gnn_input(graph, device)) - count).item()
    
    for i, budget in enumerate(budgets):
        adversarial_graph, adversarial_count= attack.find_adversarial_example(budget, adversarial_graph, subgraph, adversarial_count, similar_subgraph, increase)
        assert adversarial_count.item() == subgraph_counting(adversarial_graph, subgraph),f'true: {subgraph_counting(adversarial_graph, subgraph)}, computed: {adversarial_count.item()}'

        pred = gnn(generate_gnn_input(adversarial_graph, device))
        adversarial_prediction.append(pred)
        adversarial_errors.append((pred - count).item())    
        adversarial_counts.append(adversarial_count.item())

        
        # write down the graphs
        adversarial_graph_file = os.path.join(adversarial_graphs_folder, f"{model_architecture}_{graph_idx}_{subgraph}_{seed}_{i}.npy")
        file_content = np.array([nx.to_numpy_array(adversarial_graph), adversarial_count.item()], dtype=object)
        np.save(adversarial_graph_file, file_content) #revise
        logging.info(f'At step {i} the adversarial error is: {adversarial_errors[-1]}')
    
    # process the results
    results = {}
    results["dataset"] = os.path.splitext(os.path.basename(dataset_path))[0]
    results["graph"] = graph_idx
    results["architecture"] = model_architecture
    results["subgraph"] = subgraph
    results["seed"] = seed
    results[f"adversarial_error"] = adversarial_errors # list with avdersatial error at every budget step
    results[f"test_prediction"] = test_prediction
    results[f"adversarial_prediction"] = adversarial_prediction
    results["test_count"] = count.item()
    results["adversarial_count"] = adversarial_counts

    return results




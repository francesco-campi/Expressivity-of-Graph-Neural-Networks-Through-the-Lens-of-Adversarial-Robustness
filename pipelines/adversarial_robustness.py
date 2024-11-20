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
import shutil


sys.path.append(os.getcwd())
from src.dataset.counting_algorithm import subgraph_counting
from src.adversarial.greedy_attack import GreedyAttack, RandomGreedyAttack, PreserveGreedyAttack
from src.adversarial.beam_attack import  PreserveBeamAttack, SubstructurePreserveBeamAttack, BeamAttack
from src.metrics.L1_based import L1LossCount, L1LossStd
from src.baseline.model_gcn import GIN
from src.ppgn.ppgn import PPGN
from src.I2GNN.I2GNN import I2GNN
from src.adversarial.greedy_attack_I2GNN import I2GNNGreedyAttack, I2GNNRandomGreedyAttack
from src.adversarial.beam_attack_I2GNN import I2GNNBeamAttack, I2GNNPreserveBeamAttack, I2GNNSubstructurePreserveBeamAttack
from src.I2GNN.I2GNN_dataset import I2GNNDatasetRobustness, I2GNNDataLoader
from src.I2GNN.utils import create_subgraphs2

import logging
from sacred import Experiment
import seml
import torch_geometric.data

hops = {
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
def run(_config, dataset_path: str, graph_id: int, models_path: str, model_architecture: str, subgraph: str, seed: int, n_seeds: int, loss: str, adversarial_strategy: str, budgets: List[int], edge_addition: bool,
        edge_deletion: bool, adversarial_graphs_folder: str, device: str, n_samples: int):
    """The program gets in input one test graph to analyze the robustess and a set of models. For each model the adversarial example obtained from 
    the imput graph is computed in parallel.
    """
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:')
    logging.info(f'Dataset: {dataset_path}, models archotecture: {model_architecture}, adversarial_strategy: {adversarial_strategy}, subgraph: {subgraph}, seed: {seed}')
    experimet_id = _config['overwrite']
    root = os.path.join(os.getcwd(), f'temp_{experimet_id}')
    # initialize the adversarial attack stategy
    if model_architecture in ['GIN', 'PPGN']:
        if adversarial_strategy == "greedy":
            attack = GreedyAttack(edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        elif adversarial_strategy == "greedy_random":
            attack = RandomGreedyAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        elif adversarial_strategy == "greedy_preserve":
            attack = PreserveGreedyAttack(edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        elif adversarial_strategy == "beam":
            attack = BeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        elif adversarial_strategy == "beam_preserve":
            attack = PreserveBeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        elif adversarial_strategy == "beam_subgraph_preserving":
            attack = SubstructurePreserveBeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device)
        else:
            raise ValueError(f"The adversarial strategy {adversarial_strategy} is not supported!")
    elif model_architecture == 'I2GNN':
        if adversarial_strategy == "greedy_random":
            attack = I2GNNRandomGreedyAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device, root=root)
        elif adversarial_strategy == "beam":
            attack = I2GNNBeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device, root=root)
        elif adversarial_strategy == "beam_preserve":
            attack = I2GNNPreserveBeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device, root=root)
        elif adversarial_strategy == "beam_subgraph_preserving":
            attack = I2GNNSubstructurePreserveBeamAttack(n_samples=n_samples, edge_addition=edge_addition, edge_deletion=edge_deletion, device=device, root=root)
        else:
            raise ValueError(f"The adversarial strategy {adversarial_strategy} is not supported for i2gnn!")
    else:
        raise ValueError("The adversarial strategy is not supported!")
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
    elif model_architecture == 'I2GNN':
        gnn = I2GNN(**h_params)
    else:
        raise ValueError("The architecture is not supported!")
    gnn.load_state_dict(torch.load(model_dict, map_location=torch.device(device)))
    gnn.eval()

    # load the models for corss test
    gnns = []
    for i in range(n_seeds):
        if i != seed:
            model_dict = f"{models_path}/{model_architecture}_{subgraph}_{i}.pth"
            model_params = f"{models_path}/{model_architecture}_{subgraph}_{i}.json"
            with open(model_params, 'r') as fp:
                h_params = json.load(fp)
            if model_architecture == 'GIN':
                gnns.append(GIN(**h_params))
            elif model_architecture == 'PPGN':
                gnns.append(PPGN(**h_params))
            elif model_architecture == 'I2GNN':
                gnns.append(I2GNN(**h_params))
            else:
                raise ValueError("The architecture is not supported!")
            gnns[-1].load_state_dict(torch.load(model_dict, map_location=torch.device(device)))
            gnns[-1].eval()

    # initialize the loss
    if loss == "l1":
        loss_fn = torch.nn.L1Loss()
    elif loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss == "l1_std":
        loss_fn = L1LossStd(count_std)
    elif loss == "l1_count":
        loss_fn = L1LossCount()
    else:
        raise ValueError("Loss not supported!")

    # find adversarial example
    cross_test_errors = []
    cross_adversarial_errors = []
    cross_adversarial_errors_average = []
    adversarial_error_history = []
    adversarial_errors = []
    adversarial_prediction = []
    adversarial_counts = []
    adversarial_graph = graph
    adversarial_count = count
    sign_adversarial_errors = []
    cross_sign_adversarial_errors = []
    cross_sign_adversarial_errors_average = []
    
    if model_architecture in ['PPGN', 'GIN']:
        data = generate_gnn_input(graph, device)
        y = count
    elif model_architecture == 'I2GNN':
        dataset = I2GNNDatasetRobustness(hops[subgraph], [(graph, count)], pre_transform=pre_transform, root=root)
        dataloader = I2GNNDataLoader(dataset)
        data = next(iter(dataloader)).to(device)
        y = data.y.flatten()
    with torch.no_grad():
        test_prediction = gnn(data).item()
    # test error for the other seeds
    with torch.no_grad():
        for seed_gnn in gnns:
            cross_test_errors.append(loss_fn(seed_gnn(data).flatten(), y).item())
    
    for i, budget in enumerate(budgets):
        adversarial_graph, intermediate_adversarial_error_history, adversarial_count= attack.find_adversarial_example(budget, gnn, loss_fn, adversarial_graph, subgraph, adversarial_count)
        assert adversarial_count.item() == subgraph_counting(adversarial_graph, subgraph),f'true: {subgraph_counting(adversarial_graph, subgraph)}, computed: {adversarial_count.item()}'
        if i != 0:
            intermediate_adversarial_error_history = intermediate_adversarial_error_history[1:] # avoid repetitions
        adversarial_error_history.extend(intermediate_adversarial_error_history)
        if len(intermediate_adversarial_error_history) > 0:
            adversarial_errors.append(intermediate_adversarial_error_history[-1])
        else:
            adversarial_errors.append(adversarial_error_history[-1]) #same result as before
            
        adversarial_counts.append(adversarial_count.item())

        # cross evaluate also the other models
        intermediate_cross_adversarial_errors = []
        intermediate_cross_sign_adversarial_errors = []

        if model_architecture in ['PPGN', 'GIN']:
            data = generate_gnn_input(adversarial_graph, device)
            y = adversarial_count
        elif model_architecture == 'I2GNN':
            dataset = I2GNNDatasetRobustness(hops[subgraph], [(adversarial_graph, adversarial_count)], pre_transform=pre_transform, root=root)
            dataloader = I2GNNDataLoader(dataset)
            data = next(iter(dataloader)).to(device)
            y = data.y.flatten()

        with torch.no_grad():
            for seed_gnn in gnns:
                intermediate_cross_adversarial_errors.append(loss_fn(seed_gnn(data).flatten(), y).item())
            cross_adversarial_errors.append(intermediate_cross_adversarial_errors)
            cross_adversarial_errors_average.append(mean(intermediate_cross_adversarial_errors))
            
            adversarial_prediction.append(gnn(data).item())
            # l1 error with sign
            sign_adversarial_errors.append((gnn(data).flatten() - adversarial_count).item())
            for seed_gnn in gnns:
                intermediate_cross_sign_adversarial_errors.append((seed_gnn(data).flatten() - adversarial_count).item())
            cross_sign_adversarial_errors.append(intermediate_cross_sign_adversarial_errors)
            cross_sign_adversarial_errors_average.append(mean(intermediate_cross_sign_adversarial_errors))
        
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
    results[f"test_error"] = adversarial_error_history[0]
    results[f"adversarial_error"] = adversarial_errors # list with avdersatial error at every budget step
    results[f"sign_adversarial_errors"] = sign_adversarial_errors
    results[f"adversarial_error_history"] = adversarial_error_history
    results[f"cross_test_errors"] = cross_test_errors
    results[f"cross_test_errors_average"] = mean(cross_test_errors)
    results[f"cross_adversarial_errors"] = cross_adversarial_errors # list
    results[f"cross_sign_adversarial_errors"] = cross_sign_adversarial_errors # list
    results[f"cross_adversarial_errors_average"] = cross_adversarial_errors_average
    results[f"cross_sign_adversarial_errors_average"] = cross_sign_adversarial_errors_average
    results[f"test_prediction"] = test_prediction
    results[f"adversarial_prediction"] = adversarial_prediction
    results["test_count"] = count.item()
    results["adversarial_count"] = adversarial_counts

    if model_architecture == 'I2GNN':
        shutil.rmtree(root)

    return results




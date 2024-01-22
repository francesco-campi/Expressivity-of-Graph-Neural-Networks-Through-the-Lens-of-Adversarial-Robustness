"""Pipeline to generate the datasets for training the Graph neuarl networks on the task of substructure counting."""
import yaml
from math import ceil
from yaml import FullLoader
import os
import sys
import random
from argparse import ArgumentParser
from networkx.generators import stochastic_block_model, random_regular_graph, gnp_random_graph
from dgl import from_networkx, save_graphs
import torch
import tqdm
import multiprocessing

sys.path.append(os.getcwd())
from src.dataset.counting_algorithm import subgraph_counting_all

"""This code is ment to be run with SlurmRun or in general to be directly called form the terminal with one specific configuration file under the flag -- config."""

def generate_dataset(dataset_size, kwargs, graph_generator):
    """Returns a list of networkx graphs
    
    kwargs is alist of different configurations for the graph geenration function"""
    graphs = []
    n_configurations = len(kwargs) #number of different parameters combinations
    split_size = ceil(dataset_size / n_configurations)
    for i in range(n_configurations):
        for j in range(split_size):
            graphs.append(graph_generator(**kwargs[i]))
    return graphs # REMARK: the size might be slightly bigger than the decalred oneto keep the proportions balanced is the size and the number of configuations don't divide exactly


def generate_dataset_mix(dataset_size, kwargs, graph_generators):
    """Returns a list of networkx graphs coming from different configurations.
    kwargs and graph_generators are lists containgin graph generators and their configurations. For each element of kwargs there are several different configurations"""
    graphs = []
    n_generators = len(graph_generators)
    split_1_size = ceil(dataset_size/n_generators)
    for i in range(n_generators):
        n_configurations = len(kwargs[i])
        split_2_size = ceil(split_1_size/n_configurations)
        for j in range(n_configurations):
            for w in range(split_2_size):
                graphs.append(graph_generators[i](**kwargs[i][j]))
    print(f"Number fo generated graphs: {len(graphs)}")
    return graphs


def generate_labels(graphs, n_jobs):
    """Counts all the substructures for the given graphs and stores them in a dict"""
    # we need the keys of the dict
    counts = subgraph_counting_all(graphs[0])
    keys = list(counts.keys())
    labels = {}
    for key in keys:
        labels[key] = torch.empty((len(graphs),1))
    pool = multiprocessing.Pool(n_jobs)
    processes = [pool.apply_async(subgraph_counting_all, args=(graph,)) for graph in graphs]
    results = [p.get() for p in tqdm.tqdm(processes)] #list of dicts

    for i, result in enumerate(results):
        # update the labels
        for key in labels.keys():
            labels[key][i] = result[key]
    return labels


def main():
    # parse the input
    parser = ArgumentParser()
    parser.add_argument('--config', default='../config/training_dataset_creation.yaml')
    args = parser.parse_args()

    #read the yaml file
    with open(args.config) as f:
        config = yaml.load(f, Loader=FullLoader)
    # generate the folder
    os.makedirs(config["folder"], exist_ok = True)
    #n_jobs
    if config["n_jobs"] == -1:
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = config["n_jobs"]
    print(f"number of jobs = {n_jobs}")
    random.seed(config["seed"])
    # define the files names
    train_file = f"train_{config['dataset_size']}_{config['type']}"
    validation_file = f"validation_{config['dataset_size']}_{config['type']}"
    test_file = f"test_{config['dataset_size']}_{config['type']}"
    robustness_file = f"robustness_{config['dataset_size']}_{config['type']}"

    # generate the graphs
    if config["type"] == 'mix':
        # TODO:
        # generate graphs from all the distribution in equal proportions fro teh distributions which have parameters in the config file
        graph_generators = []
        kwargs = []
        # Erdos_Renyi
        if "probabilities" in config.keys(): #if it is not we don't use these graphs
            graph_generators.append(gnp_random_graph)
            kwargs.append([])
            for i in range(len(config["nodes"])):
                kwargs[-1].append({"n": config["nodes"][i], "p": config["probabilities"][i], "directed" : False})
        # Random Graphs
        if "d" in config.keys():
            graph_generators.append(random_regular_graph)
            kwargs.append([])
            for i in range(len(config["nodes"])):
                kwargs[-1].append({"n": config["nodes"][i], "d": config["d"][i]})
        # SBM
        if "sizes" in config.keys() and "p" in config.keys():
            graph_generators.append(stochastic_block_model)
            kwargs.append([])
            kwargs[-1] = [{"sizes": config["sizes"], "p": config["p"], "directed" : False}]
        graphs = generate_dataset_mix(config["dataset_size"], kwargs, graph_generators)
    else:
        if config["type"] == 'er':
            graph_generator = gnp_random_graph
            kwargs = []
            for i in range(len(config["nodes"])):
                kwargs.append({"n": config["nodes"][i], "p": config["probabilities"][i], "directed" : False})
                #update file names
                train_file += f"_{config['nodes'][i]}"
                validation_file += f"_{config['nodes'][i]}"
                test_file += f"_{config['nodes'][i]}"
                robustness_file += f"_{config['nodes'][i]}"
        elif config["type"] == 'rg':
            graph_generator = random_regular_graph
            kwargs = []
            for i in range(len(config["nodes"])):
                kwargs.append({"n": config["nodes"][i], "d": config["d"][i]})
                #update file names
                train_file += f"_{config['nodes'][i]}"
                validation_file += f"_{config['nodes'][i]}"
                test_file += f"_{config['nodes'][i]}"
                robustness_file += f"_{config['nodes'][i]}"
        elif config["type"] == 'sbm':
            graph_generator = stochastic_block_model
            kwargs = [{"sizes": config["sizes"], "p": config["p"], "directed" : False}]
            #update file names
            train_file += f"_{sum(config['sizes'])}"
            validation_file += f"_{sum(config['sizes'])}"
            test_file += f"_{sum(config['sizes'])}"
            robustness_file += f"_{sum(config['sizes'])}"
        else:
            raise ValueError(f"Graph type {config['type']} is not supported!")

        graphs = generate_dataset(
                config["dataset_size"],
                kwargs,
                graph_generator,
            )
    
    # might add a different options wher e the graphs are geenrated for robustness, hence only 1 dataset is generated
    if config["usage"] == "training":
        # split the graphs
        random.shuffle(graphs)
        train_graphs = graphs[:round(len(graphs)*config["train_split"])]
        print(f'Len training Graphs: {len(train_graphs)}')
        validation_graphs = graphs[round(len(graphs)*config["train_split"]):round(len(graphs)*(config["validation_split"] + config["train_split"]))]
        print(f'Len validation Graphs: {len(validation_graphs)}')
        test_graphs = graphs[-round(len(graphs)*config["test_split"]):]
        print(f'Len test Graphs: {len(test_graphs)}')
        # generate the labels
        print("Counting training lables:")
        train_labels = generate_labels(train_graphs, n_jobs)
        print("Counting validation lables:")
        validation_labels = generate_labels(validation_graphs, n_jobs)
        print("Counting test lables:")
        test_labels = generate_labels(test_graphs, n_jobs)
        # store the graphs
        dgl_train_graphs = []
        for graph in train_graphs:
            dgl_train_graphs.append(from_networkx(graph))
        if os.path.isfile(os.path.join(config["folder"], train_file + ".bin")):
            i = 2
            while os.path.isfile(os.path.join(config["folder"], train_file + f'_{i}' + ".bin")):
                i += 1
            train_file = train_file + f'_{i}'
            validation_file = validation_file + f'_{i}'
            test_file = test_file + f'_{i}'
        save_graphs(os.path.join(config["folder"], train_file + ".bin"), dgl_train_graphs, train_labels)
        dgl_validation_graphs = []
        for graph in validation_graphs:
            dgl_validation_graphs.append(from_networkx(graph))
        save_graphs(os.path.join(config["folder"], validation_file + ".bin"), dgl_validation_graphs, validation_labels)
        dgl_test_graphs = []
        for graph in test_graphs:
            dgl_test_graphs.append(from_networkx(graph))
        save_graphs(os.path.join(config["folder"], test_file + ".bin"), dgl_test_graphs, test_labels)
    
    elif config["usage"] == "robustness":
        random.shuffle(graphs)
        # generate the labels
        print("Counting lables:")
        labels = generate_labels(graphs, n_jobs)
        # store the graphs
        dgl_graphs = []
        for graph in graphs:
            dgl_graphs.append(from_networkx(graph))
        save_graphs(os.path.join(config["folder"], robustness_file + ".bin"), dgl_graphs, labels)
    
    else:
        raise ValueError(f"{config['usage']} is not supported, please use 'training' or 'robustness' instead!")


if __name__ == "__main__":
    main()
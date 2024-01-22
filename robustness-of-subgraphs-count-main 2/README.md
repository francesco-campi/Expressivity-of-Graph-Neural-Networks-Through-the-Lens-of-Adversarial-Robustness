# Robustness of subgraphs count GNNs

We perform the first adversarial robustness study into Graph Neural Networks (GNNs) that are provably more powerful than traditional Message Passing Neural Networks (MPNNs). In particular, we use adversarial robustness as a tool to uncover a significant gap between their theoretically possible and empirically achieved expressive power. To do so, we focus on the ability of GNNs to count specific subgraph patterns, which is an established measure of expressivity, and extend the concept of adversarial robustness to this task. Based on this, we develop efficient adversarial attacks for subgraph counting and show that more powerful GNNs fail to generalize even to small perturbations to the graph's structure. Expanding on this, we show that such architectures also fail to count substructures on out-of-distribution graphs.

## Required packages

|Package|Version|
|-------|-------|
|Pytorch|1.12.0|
|dgl|0.9.1|
|pyg|2.1.0|
|Numpy|1.23.1|
|Networkx|2.8.4|
|Matplotlib|3.5.2|
|SciPy|1.7.3|
|Wandb|0.13.4|
|Pandas|1.3.5|
|seml|0.3.5|



## Generate a dataset

To generate a dataset of synthetic graphs follow the instrictions on the yaml file `config/dataset_creation.yaml` and then from the repository root directory run te following command:

```sh
python pipelines/dataset_creation.py --config config/dataset_creation.yaml
```

## Train a GNNs

To train a GNNs on a previously generated datasest give the necessary parameters in the yaml file `config/*architecture*_train.yaml`  and then from the repository root directory run te following command:

```sh
seml experiment_name add config/GNN_train.yaml
seml experiment_name start --local
```

## Run adversarial attacks

To search adversarial examples on a previously trained model give the necessary parameters in the yaml file `config/*architecture*_adversarial_robustness_*dataset*.yaml`  and then from the repository root directory run te following command:

```sh
seml experiment_name add config/adversarial_robustness.yaml
seml experiment_name start --local
```

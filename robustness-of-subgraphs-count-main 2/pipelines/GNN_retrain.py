import logging
import os
from sacred import Experiment
import seml
import torch
from torch_geometric.loader import DataLoader
import time
from copy import deepcopy
import re
import sys
import wandb
from statistics import mean
import os
import json

sys.path.append(os.getcwd())
from src.baseline.model_gcn import GIN
from src.baseline.dataset_gcn import GraphDataset
from src.metrics.L1_based import L1LossCount, L1LossStd
from src.ppgn.ppgn import PPGN
from src.I2GNN.I2GNN import I2GNN
from src.I2GNN.I2GNN_dataset import I2GNNDataset, I2GNNDataLoader
from src.I2GNN.utils import create_subgraphs2

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

ex = Experiment()
seml.setup_logger(ex)
os.environ["WANDB_SILENT"] = "true"


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


def train_epoch(train_dataloader, gnn, optimizer, loss_fn, device):
    gnn.train()
    for data in train_dataloader:
        data = data.to(device)
        y = data.y

        pred = gnn(data)
        
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_epoch(dataloader, gnn, loss_fn, device)->torch.Tensor:
    gnn.eval()
    with torch.no_grad():
        num_batches = len(dataloader)
        loss = torch.zeros(1).to(device)
        for data in dataloader:
            data = data.to(device)
            y = data.y
            pred = gnn(data)
            loss += loss_fn(pred, y)
        
        loss = loss / num_batches
    return loss


@ex.automain
def run(_config, # get information about the experiment
        train_dataset: str, 
        validation_dataset: str,
        test_dataset: str,
        test_original_dataset: str,
        subgraph: str,
        model: str,
        model_folder: str,
        retrain_folder: str,
        learning_rate: float,
        max_epochs: int,
        batch_size: int,
        patience: int,
        scheduler_patience: int,
        scheduler_rate: float,
        loss: str,
        # channels: list,  
        # num_layers: int, 
        # act: str, 
        # pooling: str, 
        # aggr: str,
        # concat: bool,
        # batch_norm: bool,
        # depth_of_mlp: int,
        n_seeds: int,
    ):
    """Retrain only the final MLP of the given models on a different dataset to check if the feature extraction is sufficinetly powerful"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(os.path.join(model_folder, retrain_folder), exist_ok=True)
    experimet_id = _config['overwrite']
    nodes = train_dataset.split('/')[-1].split('.')[0].split('_')[-1]
    generative_model = train_dataset.split('/')[-1].split('.')[0].split('_')[-2]
    # wandb info
    config = {
        "learning_rate": learning_rate,  
        "batch_size": batch_size,
        "loss": loss,
    }
    project = f"{model}_{subgraph}_{loss}_{generative_model}_{nodes}_retrain"
    
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:\n')
    logging.info('Task:')
    logging.info(f'Datasets: \ntrain: {train_dataset}, \nvalidation: {validation_dataset}, \nmodel: {model}, subgraph: {subgraph}\n')
    logging.info(f'Training:\nmax epochs: {max_epochs}, learning rate: {learning_rate}, batch size: {batch_size}, patience: {patience}, device: {device}, loss function: {loss}, scheduler patience: {scheduler_patience}, scheduler rate: {scheduler_rate}')
    # logging.info(f'Hyperparams:\nin channels = {channels[0]}, hidden channels: {channels[1]}, out channles: {channels[2]},')
    # logging.info(f'num_layes: {num_layers}, activation function: {act}, pooling: {pooling},aggregation: {aggr}, concatenate: {concat}, batch normalization: {batch_norm} ')

    # load the models
    gnns = []
    for i in range(n_seeds):
        model_dict = f"{model_folder}/{model}_{subgraph}_{i}.pth"
        model_params = f"{model_folder}/{model}_{subgraph}_{i}.json"
        with open(model_params, 'r') as fp:
            h_params = json.load(fp)
        if model == 'GIN':
            gnns.append(GIN(**h_params).to(device))
        elif model == 'PPGN':
            gnns.append(PPGN(**h_params).to(device))
        elif model == 'I2GNN':
            gnns.append(I2GNN(**h_params).to(device))
        else:
            raise ValueError("The architecture is not supported!")
        gnns[-1].load_state_dict(torch.load(model_dict, map_location=torch.device(device)))

    # initialize the dataset
    if model == 'GIN' or model == 'PPGN':
        train = GraphDataset(train_dataset, subgraph, in_channels=1) #channels[0])
        validation = GraphDataset(validation_dataset, subgraph, in_channels=1) # channels[0])
        test = GraphDataset(test_dataset, subgraph, in_channels=1)
        y_train_val = torch.cat([train.labels, validation.labels], dim = 0)

    elif model == 'I2GNN':
        def pre_transform(g, hops):
            return create_subgraphs2(g, hops)
        train = I2GNNDataset(root=os.path.dirname(train_dataset),dataset=os.path.basename(train_dataset),  subgraph_type=subgraph, pre_transform=pre_transform, hops=hops[subgraph])
        validation = I2GNNDataset(root=os.path.dirname(validation_dataset),dataset=os.path.basename(validation_dataset),  subgraph_type=subgraph, pre_transform=pre_transform, hops=hops[subgraph])
        test = I2GNNDataset(root=os.path.dirname(test_dataset),dataset=os.path.basename(test_dataset),  subgraph_type=subgraph, pre_transform=pre_transform, hops=hops[subgraph])
        y_train_val = torch.cat([train.data.y, validation.data.y], dim=0)
    m = y_train_val.mean(dim=0)
    std = y_train_val.std(dim=0)
    var = y_train_val.var(dim=0)

    logging.info(f"The substructure count mean is : {m}, the variance is: {var}, the standard deviaiton is: {std}")

    # initialize the loss
    if loss == "l1":
        loss_fn = torch.nn.L1Loss()
    elif loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss == "l1_std":
        loss_fn = L1LossStd(std)
    elif loss == "l1_count":
        loss_fn = L1LossCount()
    else:
        raise ValueError("Loss not supported!")

    # initialize the final metrics
    l1 = torch.nn.L1Loss()
    l1_metrics = []
    l1_metrics_test = []

    l1_std = L1LossStd(std.to(device))
    l1_std_metrics = []
    l1_std_metrics_test = []

    l1_count = L1LossCount()
    l1_count_metrics = []
    l1_count_metrics_test = []


    model_paths = []
    h_params_paths = []

    # train one model for each seed
    for idx, seed in enumerate(range(n_seeds)):
        logging.info(f"################# Starting training with the seed number {idx}! ##############")
        torch.manual_seed(seed)

        # initialize wandb
        name = f"{experimet_id}_{idx}"
        #wandb.init(project=project, config=config, name=name)

        # initialize the dataloader
        if model == 'GIN' or model == 'PPGN':
            train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
        elif model == 'I2GNN':
            train_dataloader = I2GNNDataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = I2GNNDataLoader(dataset=validation, batch_size=batch_size, shuffle=False)
            test_dataloader = I2GNNDataLoader(dataset=test, batch_size=batch_size, shuffle=False)

        gnn = gnns[idx]

        # initialize the optimizer
        if model == 'GIN' or model == 'PPGN':
            optimizer = torch.optim.Adam(params=gnn.final_mlp.parameters(), lr=learning_rate)
        elif model == 'I2GNN':
            optimizer = torch.optim.Adam(params=[{'params' :gnn.fc1.parameters()}, 
                                     {'params' :gnn.fc2.parameters()}, 
                                     {'params' :gnn.fc3.parameters()}], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor = scheduler_rate, patience=scheduler_patience)

        # train the model for the given number of epochs and evaluate the validation performances (early sopping?)
        best_validation = float('inf')
        waitng = 0 # fro how many epoch the validation hasn't improoved
        checkpoint = gnn.state_dict() # initiaize first check point
        for epoch in range(max_epochs):
            start = time.time()
            # train one epoch
            train_epoch(train_dataloader, gnn, optimizer, loss_fn, device)
            # evaluate on training set
            train_loss = evaluate_epoch(train_dataloader, gnn, loss_fn, device).item()
            # evaluate on validation set
            validation_loss = evaluate_epoch(validation_dataloader, gnn, loss_fn, device).item()
            if epoch % 5 == 0:
                logging.info(f"Validation loss at epoch {epoch}: {validation_loss},\tTrain loss at epoch {epoch}: {train_loss},\ttime required: {time.time() - start}")
            scheduler.step(validation_loss)
            #wandb.log({"train_loss": train_loss, "validation_loss": validation_loss})
            # early stopping
            if validation_loss < best_validation:
                best_validation = validation_loss
                waitng = 0
                checkpoint = deepcopy(gnn.state_dict())
            else:
                waitng += 1
            if waitng == patience:
                break

        # reset the best model
        gnn.load_state_dict(checkpoint)
        model_file = f"{model}_{subgraph}_{idx}.pth"
        h_params_file = f"{model}_{subgraph}_{idx}.json"
        model_path = os.path.join(model_folder, retrain_folder, model_file)
        h_params_path = os.path.join(model_folder, retrain_folder, h_params_file)
        logging.info(f"The model is saved as: {model_file}, best validation loss: {best_validation}")
        torch.save(gnn.state_dict(), model_path)
        with open(h_params_path, "w") as fp:
            json.dump(h_params,fp)
        #wandb.summary["validation_loss"] = best_validation
        #wandb.finish(quiet=True)
        
        # compute the metrics
        l1_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1, device).item())
        l1_std_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1_std, device).item())
        l1_count_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1_count, device).item())

        l1_metrics_test.append(evaluate_epoch(test_dataloader, gnn, l1, device).item())
        l1_std_metrics_test.append(evaluate_epoch(test_dataloader, gnn, l1_std, device).item())
        l1_count_metrics_test.append(evaluate_epoch(test_dataloader, gnn, l1_count, device).item())

        model_paths.append(model_path)
        h_params_paths.append(h_params_path)

        


    logging.info(f"The average metrics on the validation set are: \nl1: {mean(l1_metrics)}, l1_std: {mean(l1_std_metrics)}, l1_count: {mean(l1_count_metrics)}")

    
    
    results = {
        'l1': l1_metrics, 
        'l1_average': mean(l1_metrics), 
        'l1_std': l1_std_metrics, 
        'l1_std_average': mean(l1_std_metrics), 
        'l1_count': l1_count_metrics, 
        'l1_count_average': mean(l1_count_metrics), 
        'l1_test': l1_metrics_test, 
        'l1_average_test': mean(l1_metrics_test), 
        'l1_std_test': l1_std_metrics_test, 
        'l1_std_average_test': mean(l1_std_metrics_test), 
        'l1_count_test': l1_count_metrics_test, 
        'l1_count_average_test': mean(l1_count_metrics_test), 
        'count_mean': m.item(), 
        'count_variance': var.item(), 
        'count_std': std.item(),
        'model_paths': model_paths,
        'h_param_paths': h_params_paths,
    }

    return results
import logging
import os
from sacred import Experiment
import seml
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as SANDataloader
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
from src.SAN.SAN import SAN_NodeLPE
from src.SAN.SAN_dataset import SANDataset

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


def train_epoch_SAN(train_dataloader, gnn, optimizer, loss_fn, device):
    gnn.train()
    for data in train_dataloader:
        g = data[0].to(device)
        y = data[1].to(device)

        pred = gnn(g)
        
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_epoch_SAN(dataloader, gnn, loss_fn, device)->torch.Tensor:
    gnn.eval()
    with torch.no_grad():
        num_batches = len(dataloader)
        loss = torch.zeros(1).to(device)
        for data in dataloader:
            g = data[0].to(device)
            y = data[1].to(device)
            pred = gnn(g)
            loss += loss_fn(pred, y)
        
        loss = loss / num_batches
    return loss


@ex.automain
def run(_config, # get information about the experiment
        train_dataset: str, 
        validation_dataset: str,
        test_dataset: str,
        subgraph: str,
        model: str,
        model_folder: str,
        learning_rate: float,
        max_epochs: int,
        batch_size: int,
        patience: int,
        scheduler_patience: int,
        scheduler_rate: float,
        loss: str,
        channels: list,  
        num_layers: int, 
        act: str, 
        pooling: str, 
        aggr: str,
        concat: bool,
        batch_norm: bool,
        depth_of_mlp: int,
        seeds: list,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_folder, exist_ok=True)
    experimet_id = _config['overwrite']
    nodes = train_dataset.split('/')[-1].split('.')[0].split('_')[-1]
    generative_model = train_dataset.split('/')[-1].split('.')[0].split('_')[-2]
    # wandb info
    config = {
        "learning_rate": learning_rate,  
        "batch_size": batch_size,
        "channels": channels,  
        "num_layers": num_layers, 
        "act": act, 
        "pooling": pooling,
        "aggr": aggr,
        "concat": concat,
        "batch_norm": batch_norm,
        "loss": loss,
    }
    project = f"{model}_{subgraph}_{loss}_{generative_model}_{nodes}"
    
    # Note that regularization_params contains the corresponding sub-dictionary from the configuration.
    logging.info('Received the following configuration:\n')
    logging.info('Task:')
    logging.info(f'Datasets: \ntrain: {train_dataset}, \nvalidation: {validation_dataset}, \nmodel: {model}, subgraph: {subgraph}\n')
    logging.info(f'Training:\nmax epochs: {max_epochs}, learning rate: {learning_rate}, batch size: {batch_size}, patience: {patience}, device: {device}, loss function: {loss}, scheduler patience: {scheduler_patience}, scheduler rate: {scheduler_rate}')
    logging.info(f'Hyperparams:\nin channels = {channels[0]}, hidden channels: {channels[1]}, out channles: {channels[2]},')
    logging.info(f'num_layes: {num_layers}, activation function: {act}, pooling: {pooling},aggregation: {aggr}, concatenate: {concat}, batch normalization: {batch_norm} ')

    
    # initialize the dataset
    if model == 'GIN' or model == 'PPGN':
        train = GraphDataset(train_dataset, subgraph, in_channels=channels[0])
        validation = GraphDataset(validation_dataset, subgraph, in_channels=channels[0])
        y_train_val = torch.cat([train.labels, validation.labels], dim = 0)

    elif model == 'I2GNN':
        def pre_transform(g, hops):
            return create_subgraphs2(g, hops)
        train = I2GNNDataset(root=os.path.dirname(train_dataset),dataset=os.path.basename(train_dataset),  subgraph_type=subgraph, pre_transform=pre_transform, hops=hops[subgraph])
        validation = I2GNNDataset(root=os.path.dirname(validation_dataset),dataset=os.path.basename(validation_dataset),  subgraph_type=subgraph, pre_transform=pre_transform, hops=hops[subgraph])
        y_train_val = torch.cat([train.data.y, validation.data.y], dim=0)
    
    elif model == 'SAN':
        train = SANDataset(dataset_path=train_dataset, subgraph=subgraph, in_channels=channels[0])
        validation = SANDataset(dataset_path=validation_dataset, subgraph=subgraph, in_channels=channels[0])
        y_train_val = torch.tensor([d[1].item() for d in train.data] + [d[1].item() for d in train.data])

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
    l1_std = L1LossStd(std.to(device))
    l1_std_metrics = []
    l1_count = L1LossCount()
    l1_count_metrics = []

    model_paths = []
    h_params_paths = []

    # train one model for each seed
    for idx, seed in enumerate(seeds):
        logging.info(f"################# Starting training with the seed number {idx}! ##############")
        torch.manual_seed(seed)

        # initialize wandb
        name = f"{experimet_id}_{idx}"
        #wandb.init(project=project, config=config, name=name)

        # initialize the dataloader
        if model == 'GIN' or model == 'PPGN':
            train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)
        elif model == 'I2GNN':
            train_dataloader = I2GNNDataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = I2GNNDataLoader(dataset=validation, batch_size=batch_size, shuffle=False)
        elif model == 'SAN':
            train_dataloader = SANDataloader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=train.collate)
            validation_dataloader = SANDataloader(dataset=validation, batch_size=batch_size, shuffle=False, collate_fn=validation.collate)
        # initialize the model
        h_params = {}
        if model == 'GIN':
            h_params = {
                'in_channels': channels[0], 
                'hidden_channels': channels[1], 
                'num_layers': num_layers, 
                'out_channels': channels[2], 
                'act': act, # make it a string 
                'pooling': pooling, 
                'aggr': aggr, 
                'concat': concat, 
                'batch_norm': batch_norm,
            }
            gnn = GIN(**h_params).to(device)
        elif model == 'PPGN':
            h_params = {
                'in_channels': channels[0], 
                'hidden_channels': channels[1], 
                'num_layers': num_layers, 
                'out_channels': channels[2], 
                'depth_of_mlp': depth_of_mlp,
                'act': act, # make it a string 
                'pooling': pooling,  
                'concat': concat, 
            }
            gnn = PPGN(**h_params).to(device)
        elif model == 'I2GNN':
            h_params = {
                'num_layers': num_layers
            }
            gnn = I2GNN(**h_params).to(device)
        elif model == 'SAN':
            h_params = {
                'hidden_channels': channels[1], 
                'num_layers': num_layers, 
                'out_channels': channels[2], 
                'act': act, # make it a string 
                'pooling': pooling,  
                'concat': concat, 
                'batch_norm': batch_norm,
            }
            gnn = SAN_NodeLPE(**h_params).to(device)
        else:
            raise ValueError("Model not supported!")

        # initialize the optimizer
        optimizer = torch.optim.Adam(params=gnn.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor = scheduler_rate, patience=scheduler_patience)

        # train the model for the given number of epochs and evaluate the validation performances (early sopping?)
        best_validation = float('inf')
        waitng = 0 # fro how many epoch the validation hasn't improoved
        checkpoint = gnn.state_dict() # initiaize first check point
        for epoch in range(max_epochs):
            start = time.time()
            # train one epoch
            if model == 'SAN':
                train_epoch_SAN(train_dataloader, gnn, optimizer, loss_fn, device)
                train_loss = evaluate_epoch_SAN(train_dataloader, gnn, loss_fn, device).item()
                validation_loss = evaluate_epoch_SAN(validation_dataloader, gnn, loss_fn, device).item()

            else:
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
        model_file = f"{model}_{subgraph}_{experimet_id}_{idx}.pth"
        h_params_file = f"{model}_{subgraph}_{experimet_id}_{idx}.json"
        model_path = os.path.join(model_folder, model_file)
        h_params_path = os.path.join(model_folder, h_params_file)
        logging.info(f"The model is saved as: {model_file}, best validation loss: {best_validation}")
        torch.save(gnn.state_dict(), model_path)
        with open(h_params_path, "w") as fp:
            json.dump(h_params,fp)
        #wandb.summary["validation_loss"] = best_validation
        #wandb.finish(quiet=True)
        
        # compute the metrics
        if model == 'SAN':
            l1_metrics.append(evaluate_epoch_SAN(validation_dataloader, gnn, l1, device).item())
            l1_std_metrics.append(evaluate_epoch_SAN(validation_dataloader, gnn, l1_std, device).item())
            l1_count_metrics.append(evaluate_epoch_SAN(validation_dataloader, gnn, l1_count, device).item())
        else:
            l1_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1, device).item())
            l1_std_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1_std, device).item())
            l1_count_metrics.append(evaluate_epoch(validation_dataloader, gnn, l1_count, device).item())

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
        'count_mean': m.item(), 
        'count_variance': var.item(), 
        'count_std': std.item(),
        'model_paths': model_paths,
        'h_param_paths': h_params_paths,
    }

    return results


    """pattern = f"^{model}_{subgraph}.*"
    previsous_best = None
    for file in os.listdir(model_folder):
        if re.match(pattern, file):
            previsous_best = file
            break
    if previsous_best:
        best_gnn = torch.load(os.path.join(model_folder, previsous_best)).to(device)
        previsous_best_validation = evaluate_epoch(validation_dataloader, best_gnn, loss_fn, device)
        store = (previsous_best_validation > best_validation)
    
    # store the model if it has better performances than the prevoius best
    if previsous_best is None or store:
        os.remove(os.path.join(model_folder, previsous_best))
        model_file = f"{model}_{subgraph}_{time.time()}.pth"
        model_path = os.path.join(model_folder, model_file)
        logging.info(f"The model is saved as: {model_file}")
        torch.save(checkpoint, model_path)
    else:
        logging.info("Prevois model is better.")"""

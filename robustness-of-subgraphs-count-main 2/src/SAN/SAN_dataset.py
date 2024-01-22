import torch
import pickle
import torch.utils.data
import time
import numpy as np


import dgl
import torch.nn.functional as F


from scipy import sparse as sp
import numpy as np
import networkx as nx

def laplace_decomp(g, max_freqs):


    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g



def make_full_graph(g):

    
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    return full_g


def add_edge_laplace_feats(g):

    
    EigVals = g.ndata['EigVals'][0].flatten()
    
    source, dest = g.find_edges(g.edges(form='eid'))
    
    #Compute diffusion distances and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs']-g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(),1).unsqueeze(2)
    
    
    #No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']
    
    return g



class SANDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str, subgraph: str, in_channels: int = 0):
        """
            Loading  dataset
        """
        self.subgraph = subgraph
        self.dataset_path = dataset_path
        self.in_channels = in_channels
        self.m = 0
        graphs, labels = dgl.load_graphs(self.dataset_path)
        self.labels: torch.Tensor = labels[self.subgraph]
        assert len(graphs) == len(self.labels), f"Number fo graphs and number of labels don't match: number of graphs is {len(graphs)} while number of labels is {len(self.labels)}"
        self.data = []
        for graph, count in zip(graphs, self.labels):
            if graph.number_of_nodes() > self.m:
                self.m = graph.number_of_nodes()
            graph.ndata['feat'] = torch.ones((graph.num_nodes()), dtype=torch.long)
            graph.edata['feat'] = torch.ones((graph.num_edges()), dtype=torch.long)
            self.data.append([graph, count])
        self._laplace_decomp()
        self._make_full_graph()


    def collate(self, samples):
        graphs = []
        labels = []
        for sample in samples:
            graphs.append(sample[0])
            labels.append(sample[1].item())
        labels = torch.tensor(np.array(labels), dtype=torch.long).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
    

    def _laplace_decomp(self):
        self.data = [[laplace_decomp(d[0], self.m), d[1]] for d in self.data]       

    def _make_full_graph(self):
        self.data = [[make_full_graph(d[0]), d[1]] for d in self.data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
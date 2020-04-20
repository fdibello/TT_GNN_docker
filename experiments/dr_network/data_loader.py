import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math




class SingleJetDataset(Dataset):
    def __init__(self, df,var_transformations):
        self.df = df.copy()
        
        
        self.n_jets = len(self.df)
        
        self.n_neighbors = self.df.n_neighbors
    
        for col in ['jet_pt','jet_eta','jet_phi']:
            mean, std = var_transformations[col]['mean'], var_transformations[col]['std']
            self.df[col] = (self.df[col].values-mean)/std
        
        drmean,dr_std = var_transformations['dR']['mean'], var_transformations['dR']['std']
        
        self.dRs = (self.df['dR'].values-drmean)/dr_std
        self.flavs = self.df['dR_flav'].values
        
        self.xs = self.df[['jet_pt','jet_eta','jet_phi','jet_label']].values

        self.ys = self.df.jet_tag.values

    def __len__(self):
       
        return self.n_jets


    def __getitem__(self, idx):
        
        x = torch.FloatTensor(self.xs[idx])
        
        neighbor_block = torch.FloatTensor( np.column_stack((self.dRs[idx],self.flavs[idx]) ))
        y = torch.FloatTensor([self.ys[idx]])


        return x, neighbor_block, y


class JetEventsSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the events
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]
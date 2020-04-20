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


class JetEventsDataset(Dataset):
    def __init__(self, df,var_transformations):
        self.df = df.copy()
        
        self.event_numbers = list(set(self.df['entry'].values))
        self.n_events = len(self.event_numbers)
        
        min_event = np.amin(self.event_numbers)
        
        self.event_njets = np.histogram( self.df.entry.values ,
                                        bins=np.linspace(-0.5+min_event,min_event+self.n_events+0.5-1,self.n_events+1))[0]
        
        for col in ['jet_pt','jet_eta','jet_phi']:#,'jet_label']:
            mean, std = var_transformations[col]['mean'],  var_transformations[col]['std']
            self.df[col] = (self.df[col].values-mean)/std
        
        all_xs = self.df[['jet_pt','jet_eta','jet_phi','jet_label']].values
        
        splitindices = []
        running_sum = 0
        for idx in self.event_njets:
            splitindices.append(idx+running_sum)
            running_sum+=idx
        
        self.xs = np.split(all_xs,splitindices)

        self.xs = [torch.FloatTensor(x) for x in self.xs]

        self.ys = np.split(self.df.jet_tag.values,splitindices)
        self.tagged = self.ys
        self.ys = [torch.FloatTensor(x) for x in self.ys]

        self.jet_labels = np.split(self.df.jet_label.values,splitindices)

        self.jet_labels = [torch.FloatTensor(x) for x in self.jet_labels]
        self.tagged = [torch.FloatTensor(x) for x in self.tagged]
        
    def __len__(self):
       
        return self.n_events


    def __getitem__(self, idx):
        
        x,y = self.xs[idx],self.ys[idx]
        label = self.jet_labels[idx]
        tagged = self.tagged[idx]
        return x, y,label,tagged


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
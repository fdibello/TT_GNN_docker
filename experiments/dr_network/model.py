import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

import math
import itertools

class JetEfficiencyNet(nn.Module):
    def __init__(self, in_features, hidden_layers, neighbor_feats, neighbor_layer_sizes):
        
        super(JetEfficiencyNet, self).__init__()

        jet_eff_layers = []
        jet_eff_layers.append(nn.Linear(in_features,hidden_layers[0]))
        jet_eff_layers.append(nn.ReLU())
        
        for hidden_i in range(1,len(hidden_layers)):
            jet_eff_layers.append(nn.Linear(hidden_layers[hidden_i-1],hidden_layers[hidden_i]))
            jet_eff_layers.append(nn.ReLU())
        
        jet_eff_layers.append(nn.Linear(hidden_layers[-1],1))
        jet_eff_layers.append(nn.Sigmoid())
                
        self.jet_eff = nn.Sequential( *jet_eff_layers )
    
        neighbor_layers = []
        neighbor_layers.append(nn.Linear(neighbor_feats+in_features,neighbor_layer_sizes[0]))
        neighbor_layers.append(nn.ReLU())
        for hidden_i in range(1,len(neighbor_layer_sizes)):
            neighbor_layers.append(nn.Linear(neighbor_layer_sizes[hidden_i-1],neighbor_layer_sizes[hidden_i]))
            neighbor_layers.append(nn.ReLU())
        neighbor_layers.append(nn.Linear(neighbor_layer_sizes[-1],1))
        neighbor_layers.append(nn.Sigmoid())
        
        self.neighbor_eff = nn.Sequential( *neighbor_layers )
        
    def forward(self, jet_inp, neighbor_inp):
        
        B, n_neighbor, _ = neighbor_inp.shape
        
        jet_eff =  self.jet_eff(jet_inp) 
        
        block = torch.cat( (jet_inp.unsqueeze(1).repeat(1,n_neighbor,1),neighbor_inp),dim=2 )
        
        neighbor_eff =  self.neighbor_eff(block) 
        
        neighbor_correction = torch.prod(neighbor_eff,1)
        
        

        return torch.mul(jet_eff,neighbor_correction)
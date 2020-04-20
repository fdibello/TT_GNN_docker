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


class Attention(nn.Module):
    def __init__(self, in_features,outfeatures):
        super().__init__()
#         small_in_features = max(math.floor(in_features/10), 5)
#         self.d_k = small_in_features
        
        midfeatures = int((2*in_features+outfeatures)/2)
        self.value = nn.Sequential(
                nn.Linear(2*in_features,midfeatures),
                nn.ReLU(),
                nn.Linear(midfeatures,outfeatures),
                nn.Tanh())
#         self.query = nn.Sequential(
#             nn.Linear(in_features, small_in_features),
#             nn.Tanh(),
#         )
#         self.key = nn.Linear(in_features, small_in_features)

    def forward(self, inp):
        # inp.shape should be (B,N,C)
#         q = self.query(inp)  # (B,N,C/10)
#         k = self.key(inp)  # B,N,C/10
        
        B, n, _ = inp.shape
        m2 = inp.repeat(1, n, 1)[:,np.delete(np.arange(n*n),[i*n+i for i in range(n)]),:]
        m1 = inp[:,np.repeat(np.arange(n),n-1),:]
        block = torch.cat([m1,m2],dim=2)
        block = block.view(B,n,n-1,-1)
        
        value = self.value(block) # B,N,N-1
        
#         x = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # B,N,N

#         x = x.transpose(1, 2)  # (B,N,N)
#         x = x.softmax(dim=2)  # over rows
#         x = torch.matmul(x, value)  # (B, N, outfeatures)
        return torch.sum(value,dim=2) #x

class JetEfficiencyNet(nn.Module):
    def __init__(self, in_features, feats,node_eff_layers,correction_layers):

        super(JetEfficiencyNet, self).__init__()

        self.layers = nn.ModuleList([])

        self.layers.append(DeepSetLayer(in_features,0, feats[0]))
        for i in range(1, len(feats)):
            
            self.layers.append(DeepSetLayer(in_features,feats[i-1], feats[i]))

        self.n_layers = len(self.layers)
        self.activ = nn.Tanh()
        
        jet_eff_layers = []
        jet_eff_layers.append(nn.Linear(in_features,node_eff_layers[0]))
        jet_eff_layers.append(nn.ReLU())
            
        for hidden_i in range(1,len(node_eff_layers)):
            jet_eff_layers.append(nn.Linear(node_eff_layers[hidden_i-1],node_eff_layers[hidden_i]))
            jet_eff_layers.append(nn.ReLU())
            
        jet_eff_layers.append(nn.Linear(node_eff_layers[-1],1))
        jet_eff_layers.append(nn.Sigmoid())
        
        self.jet_eff = nn.Sequential( *jet_eff_layers )
        
        eff_correction_layers = []
        eff_correction_layers.append(nn.Linear(in_features+feats[-1],correction_layers[0]))
        eff_correction_layers.append(nn.ReLU())
            
        for hidden_i in range(1,len(correction_layers)):
            eff_correction_layers.append(nn.Linear(correction_layers[hidden_i-1],correction_layers[hidden_i]))
            eff_correction_layers.append(nn.ReLU())
            
        eff_correction_layers.append(nn.Linear(correction_layers[-1],1))
        eff_correction_layers.append(nn.Sigmoid())
        
        self.eff_correction = nn.Sequential( *eff_correction_layers )
        
    def forward(self, inp):
        effs = self.jet_eff(inp)
        
        x = inp
        for layer_i in range(self.n_layers):
            x = self.layers[layer_i](x)
            #if layer_i < self.n_layers-1:
            #x = self.activ(x)
            x = torch.cat((inp,x),dim=2)
        
        correction = self.eff_correction(x)
        out = torch.mul(effs,correction)
        return out


class DeepSetLayer(nn.Module):
    def __init__(self, original_in_features, in_features, out_features):
        
        super(DeepSetLayer, self).__init__()
        
        out_features = int(out_features/2)
        mid_features = int((original_in_features+in_features+out_features)/2)
        
        self.attention = Attention(original_in_features+in_features,out_features)
        self.layer1 = nn.Sequential(
            nn.Conv1d(original_in_features+in_features, mid_features, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(mid_features, out_features, 1, bias=True),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(out_features, out_features, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(out_features, out_features, 1, bias=True),
            nn.Tanh()
        )
        #self.layer2 = nn.Conv1d(original_in_features+in_features, out_features, 1, bias=True)

    def forward(self, inp):
        # x.shape = (B,N,C)
        
        x_T = inp.transpose(2, 1)  # B,N,C -> B,C,N
          
        attention = self.attention(inp).transpose(2,1)
        
        x = torch.cat([self.layer1(x_T), self.layer2(attention)],dim=1)
        
        # normalization
        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
        
        x = x.transpose(1, 2) # B,C,N -> B,N,C
        
        return x
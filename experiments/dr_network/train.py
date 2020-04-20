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
import json

from helper_functions import *
from model import *
from data_loader import *


f = uproot.open('/Users/jshlomi/Desktop/Datasets/TruthTagging/datasetNewFormat_800k.root')

n_entries = f['t1'].numentries
training_split = 600000



jets_train_data = collect_jets(f,0,training_split)
jet_valid_data = collect_jets(f,training_split,n_entries)


jets_train_data['dR'], jets_train_data['dR_flav'], jets_train_data['n_neighbors'] = compute_dRs(jets_train_data)
jet_valid_data['dR'], jet_valid_data['dR_flav'], jet_valid_data['n_neighbors'] = compute_dRs(jet_valid_data)


var_transformations = {}
for var_i,var  in enumerate(['jet_pt','jet_eta','jet_phi']):
    var_values = jets_train_data[var].values
    var_transformations[var] = {'mean' : np.mean(var_values), 'std' : np.std(var_values)}

dR_values = np.concatenate(jets_train_data['dR'].values)
var_transformations['dR'] = {'mean' : np.mean(dR_values), 'std' : np.std(dR_values)}


with open('var_transformations.json', 'w') as fp:
	converted_var_transformations = {}
	for key in var_transformations:
		converted_var_transformations[key] = {}
		for val in ['mean','std']:
			converted_var_transformations[key][val] = str(var_transformations[key][val])

	json.dump(converted_var_transformations, fp)

b_jet_ds = SingleJetDataset(jets_train_data[jets_train_data.jet_label==1],var_transformations)
c_jet_ds = SingleJetDataset(jets_train_data[jets_train_data.jet_label==2],var_transformations)
u_jet_ds = SingleJetDataset(jets_train_data[jets_train_data.jet_label==3],var_transformations)


batch_size = 1000
batch_sampler_bjets = JetEventsSampler(b_jet_ds.n_neighbors,batch_size)
data_loader_bjets = DataLoader(b_jet_ds,batch_sampler=batch_sampler_bjets)

batch_sampler_cjets = JetEventsSampler(c_jet_ds.n_neighbors,batch_size)
data_loader_cjets = DataLoader(c_jet_ds,batch_sampler=batch_sampler_cjets)

batch_sampler_ujets = JetEventsSampler(u_jet_ds.n_neighbors,batch_size)
data_loader_ujets = DataLoader(u_jet_ds,batch_sampler=batch_sampler_ujets)


b_net = JetEfficiencyNet(4,[50,50],2,[50,50])
c_net = JetEfficiencyNet(4,[50,50],2,[50,50])
light_net = JetEfficiencyNet(4,[50,50],2,[50,50])



loss_vs_epoch = []


lossfunc = nn.BCELoss()
b_optimizer = optim.Adam(b_net.parameters(), lr=0.001)
c_optimizer = optim.Adam(c_net.parameters(), lr=0.001)
light_optimizer = optim.Adam(light_net.parameters(), lr=0.001)


epochs = 2
#net.train()

if len(loss_vs_epoch) == 0:
    first_epoch = 0
else:
    first_epoch = len(loss_vs_epoch)

def train_dataset(d_loader, net, optimizer):
    net.train()
    batch_losses = []
    for x,dRbatch,y in tqdm(d_loader):
        optimizer.zero_grad()
        
        output = net(x,dRbatch)
        
        loss = lossfunc(output,y)
        batch_losses.append(loss.item())
        
        loss.backward()  
        optimizer.step()
    return np.mean(batch_losses)
    
    
for epoch in range(first_epoch,first_epoch+epochs):
    
    b_loss = train_dataset(data_loader_bjets, b_net, b_optimizer)
    c_loss = train_dataset(data_loader_cjets, c_net, c_optimizer)
    light_loss = train_dataset(data_loader_ujets, light_net, light_optimizer)
    loss_vs_epoch.append([epoch,b_loss,c_loss,light_loss])
    

    print(loss_vs_epoch[-1])

torch.save(b_net.state_dict(), 'b_net.pt')
torch.save(c_net.state_dict(), 'c_net.pt')
torch.save(light_net.state_dict(), 'light_net.pt')

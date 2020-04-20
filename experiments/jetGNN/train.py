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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


f = uproot.open('../dataset5M.root')

n_entries = f['t1'].numentries
training_split = 3000000



jets_train_data = collect_jets(f,0,training_split)
#jet_valid_data = collect_jets(f,training_split,n_entries)


var_transformations = {}
for var_i,var  in enumerate(['jet_pt','jet_eta','jet_phi']):
    var_values = jets_train_data[var].values
    var_transformations[var] = {'mean' : np.mean(var_values), 'std' : np.std(var_values)}

with open('var_transformations.json', 'w') as fp:
	converted_var_transformations = {}
	for key in var_transformations:
		converted_var_transformations[key] = {}
		for val in ['mean','std']:
			converted_var_transformations[key][val] = str(var_transformations[key][val])

	json.dump(converted_var_transformations, fp)

ds = JetEventsDataset(jets_train_data,var_transformations)



batch_size = 5000
batch_sampler = JetEventsSampler(ds.event_njets,batch_size)
data_loader= DataLoader(ds,batch_sampler=batch_sampler)


nets = []
n_nets = 20
for _ in range(n_nets):
    nets.append( JetEfficiencyNet(4,[256,256,256,256],[256,128,50]) )

loss_vs_epoch = []


lossfunc = nn.BCELoss()
light_loss = nn.BCELoss(reduction='sum')
optimizers = [ optim.Adam(nets[i].parameters(), lr=0.001)  for i in range(n_nets)]

light_jet_factor = 1

epochs = 42

for net in nets:
    net.train()
    net.cuda()

if len(loss_vs_epoch) == 0:
    first_epoch = 0
else:
    first_epoch = len(loss_vs_epoch)

for net, optimizer in zip(nets,optimizers):
    for epoch in range(first_epoch,first_epoch+epochs):
        if epoch == 20:
            optimizers = [ optim.Adam(nets[i].parameters(), lr=0.0005)  for i in range(n_nets)]
        if epoch == 40:
            optimizers = [ optim.Adam(nets[i].parameters(), lr=0.0001)  for i in range(n_nets)]
        batch_losses = []
        for x,y,label,tagged_untagged in tqdm(data_loader):
            x = x.cuda()
            y = y.cuda()
            label = label.data.numpy()
            tagged_untagged = tagged_untagged.data.numpy()
            

            b_mask = np.where(label.flatten()==1)
            c_mask = np.where(label.flatten()==2)
            u_tagged_mask = np.where( (label.flatten()==3) & (tagged_untagged.flatten()==1) )
            u_untagged_mask =np.where( (label.flatten()==3) & (tagged_untagged.flatten()==0) )
            n_tagged_ujets = len(u_tagged_mask[0])
            n_untagged_ujets = len(u_untagged_mask[0])
            n_light_jets = n_tagged_ujets+light_jet_factor*n_untagged_ujets

            
            optimizer.zero_grad()
        
            output = net(x)
        
            output = output.view(-1)
            y = y.view(-1)
        
            
            b_loss = lossfunc(output[b_mask],y[b_mask])    
            c_loss = lossfunc(output[c_mask],y[c_mask])
        
            u_loss = light_loss(output[u_tagged_mask],y[u_tagged_mask])+light_jet_factor*light_loss(output[u_untagged_mask],y[u_untagged_mask])
            u_loss = u_loss/n_light_jets

            loss = c_loss+b_loss+u_loss #lossfunc(output.view(-1),y.view(-1))
            batch_losses.append([loss.item(),b_loss.item(),c_loss.item(),u_loss.item()])
        
            loss.backward()  
            optimizer.step()
        batch_losses = np.array(batch_losses)
        loss_vs_epoch.append([epoch]+[np.mean(batch_losses[:,i]) for i in range(4)])
        print(loss_vs_epoch[-1])


for net_i, net in enumerate(nets):
    torch.save(net.state_dict(), 'net_'+str(net_i)+'.pt')


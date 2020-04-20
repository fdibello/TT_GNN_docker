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
import sys

from helper_functions import *
from model import *
from data_loader import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#f = uproot.open('../dataset10M_sameFlav.root')
f = uproot.open(sys.argv[1])

n_entries = f['t1'].numentries


jets_test_data = collect_jets(f,0,n_entries) 



n_nets = 20
nets = []
for i in range(n_nets):
    nets.append( JetEfficiencyNet(4,[256,256,256,256],[256,128,50]) )

    nets[-1].load_state_dict( torch.load('net_'+str(i)+'.pt',map_location='cpu'))


    nets[-1].cuda()
    nets[-1].eval();

with open('var_transformations.json', 'r') as fp:
    var_transformations = json.load(fp)

for key in var_transformations:
	for val in ['mean','std']:
		var_transformations[key][val] = float(var_transformations[key][val])

test_ds = JetEventsDataset(jets_test_data,var_transformations)

def eval_on_ds(nets, ds):
    
    
    indx_list = []
    predictions = [[] for _ in range(n_nets)]
    

    for njet_in_event in tqdm( range(2,np.amax(ds.event_njets)+1) ):

        event_indxs = np.where(ds.event_njets == njet_in_event)[0]

        indx_list+=list(event_indxs)

        n_sub_batches = len(event_indxs)/10000

        sub_batches = np.array_split(event_indxs, n_sub_batches)
        sub_predictions = [[] for _ in range(n_nets)]

        for sub_batch in tqdm(sub_batches):
            input_batch = np.stack([ds[i][0] for i in sub_batch])
            
            for net_i, net in enumerate(nets):
                output =  net( torch.tensor( input_batch ).cuda() ).cpu().data.numpy()
                sub_predictions[net_i].append(output)
        sub_predictions = [ np.concatenate(sub_predictions[i]) for i in range(n_nets)]
        for net_i in range(n_nets):
            predictions[net_i]+=list( sub_predictions[net_i] )


    sorted_predictions = [ [x for _, x in sorted(zip(indx_list, predictions[net_i])) ] for net_i in range(n_nets) ]
    
    return sorted_predictions

sorted_predictions = eval_on_ds(nets,test_ds)
#print('----')
#print(sorted_predictions[0].shape,sorted_predictions[1].shape,sorted_predictions[2].shape)
sorted_predictions = [ np.concatenate(sorted_predictions[net_i]) for net_i in range(n_nets)  ]
#print(len(jets_test_data),len(sorted_predictions))


np.save('prediction_'+sys.argv[2]+'.npy',np.column_stack(sorted_predictions))
#np.save('prediction.npy',np.column_stack(sorted_predictions))
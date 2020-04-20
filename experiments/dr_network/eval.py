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



f = uproot.open('/Users/jshlomi/Desktop/Datasets/TruthTagging/dataset5M.root')


n_entries = f['t1'].numentries


jets_test_data = collect_jets(f,0,n_entries)

jets_test_data['dR'], jets_test_data['dR_flav'], jets_test_data['n_neighbors'] = compute_dRs(jets_test_data)


b_net = JetEfficiencyNet(4,[50,50],2,[50,50])
c_net = JetEfficiencyNet(4,[50,50],2,[50,50])
light_net = JetEfficiencyNet(4,[50,50],2,[50,50])

b_net.load_state_dict( torch.load('b_net.pt',map_location='cpu'))
c_net.load_state_dict( torch.load('c_net.pt',map_location='cpu'))
light_net.load_state_dict( torch.load('light_net.pt',map_location='cpu'))


with open('var_transformations.json', 'r') as fp:
    var_transformations = json.load(fp)

for key in var_transformations:
	for val in ['mean','std']:
		var_transformations[key][val] = float(var_transformations[key][val])

test_ds = SingleJetDataset(jets_test_data,var_transformations)

def eval_on_ds(net, ds):
    
    net.eval();
    indx_list = []
    predictions = []
    

    for njet_neighbors in tqdm( range(1,np.amax(ds.n_neighbors)+1) ):

        jet_indxs = np.where(ds.n_neighbors == njet_neighbors)[0]
        if len(jet_indxs) < 1:
            continue
        indx_list+=list(jet_indxs)

        input_batch = np.stack([ds[i][0] for i in jet_indxs])
        input_neighbors_batch = np.stack([ds[i][1] for i in jet_indxs])
        
        output =  net( torch.tensor( input_batch ), torch.tensor(input_neighbors_batch) ).data.numpy()

        predictions+=list( output )


    sorted_predictions = [x[0] for _, x in sorted(zip(indx_list, predictions))]
    
    return sorted_predictions

jets_test_data['b_eff'] = eval_on_ds(b_net,test_ds)
jets_test_data['c_eff'] = eval_on_ds(c_net,test_ds)
jets_test_data['u_eff'] = eval_on_ds(light_net,test_ds)


jets_test_data['NN_predictions'] = np.zeros(len(jets_test_data))

jets_test_data.loc[jets_test_data.jet_label==1,'NN_predictions'] = jets_test_data.loc[jets_test_data.jet_label==1]['b_eff']
jets_test_data.loc[jets_test_data.jet_label==2,'NN_predictions'] = jets_test_data.loc[jets_test_data.jet_label==2]['c_eff']
jets_test_data.loc[jets_test_data.jet_label==3,'NN_predictions'] = jets_test_data.loc[jets_test_data.jet_label==3]['u_eff']



np.save('prediction.npy',jets_test_data['NN_predictions'].values)
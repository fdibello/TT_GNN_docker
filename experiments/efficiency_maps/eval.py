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
import pickle

from helper_functions import *




f = uproot.open('/Users/jshlomi/Desktop/Datasets/TruthTagging/datasetB.root')

with open('effmaps.pkl', 'rb') as fp:
    eff_maps = pickle.load(fp)

n_entries = f['t1'].numentries


jets_test_data = collect_jets(f,0,n_entries)

pt_bins = np.linspace(20,600,20)
eta_bins = np.linspace(-2.5,2.5,11)

def apply_effmaps(ds):

    pt_values = ds['jet_pt'].values
    eta_values = ds['jet_eta'].values

    results_effs = np.zeros(len(pt_values))

    for flav in [1,2,3]:
        flav_cut = np.where( ds['jet_label'].values==flav )[0]

        bins = [ 
                [np.min([x,len(pt_bins)-1])-1 for x in np.digitize(pt_values[flav_cut],pt_bins)],
              [np.min([x,len(eta_bins)-1])-1 for x in np.digitize(eta_values[flav_cut],eta_bins)] 
        ]

        effs = eff_maps[flav][tuple(bins)]

        results_effs[flav_cut] = effs

    
    ds['map_eff'] = results_effs

apply_effmaps(jets_test_data)



np.save('prediction.npy',jets_test_data['map_eff'].values)
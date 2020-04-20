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



f = uproot.open('/Users/jshlomi/Desktop/Datasets/TruthTagging/dataset5M.root')

n_entries = f['t1'].numentries
training_split = 5000000



jets_train_data = collect_jets(f,0,training_split)
#jet_valid_data = collect_jets(f,training_split,n_entries)


eff_maps = {
    1: None,
    2: None,
    3: None,
}

for flav_i, flav in enumerate([1,2,3]):
    flav_cut = np.where(jets_train_data['jet_label']==flav)[0]
    
    pt_values = jets_train_data['jet_pt'].values[flav_cut]
    eta_values = jets_train_data['jet_eta'].values[flav_cut]
    istagged_values = jets_train_data['jet_tag'].values[flav_cut]

    pt_bins = np.linspace(20,600,20)
    eta_bins = np.linspace(-2.5,2.5,11)

    tagged_jets = np.where(istagged_values ==1)[0]
    total_histogram = np.histogram2d(pt_values,eta_values,bins=(pt_bins,eta_bins))
    pass_histogram = np.histogram2d(pt_values[tagged_jets],eta_values[tagged_jets],bins=(pt_bins,eta_bins))

    eff_map = np.divide(pass_histogram[0],total_histogram[0])

    eff_maps[flav] = eff_map.copy()

f = open("effmaps.pkl","wb")
pickle.dump(eff_maps,f)
f.close()



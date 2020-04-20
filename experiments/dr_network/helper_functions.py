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

def collect_jets(f,entrystart,entry_stop):
    df = f['t1'].pandas.df(['jet_pt','jet_eta','jet_phi','jet_label','jet_eff','jet_score'],
                           entrystart=entrystart,entrystop=entry_stop).reset_index()
    df['jet_tag'] = 1*(df.jet_score < df.jet_eff)
    return df




def deltaphi(x):
    
    
    while len(np.where(x >= np.pi)[0]) > 0:
        x[np.where(x >= np.pi)[0]] -= 2*np.pi;
    while len(np.where(x < -np.pi)[0]) > 0: 
        x[np.where(x < -np.pi)[0]]+= 2*np.pi
    return x;

def deltaR(eta,phi,eta2,phi2):
    deta = eta-eta2
    dphi = deltaphi(phi-phi2)
    return np.sqrt( deta*deta+dphi*dphi )

def compute_dRs(df):
    event_numbers = list(set(df['entry'].values))
    n_events = len(event_numbers)
    min_event = np.amin(event_numbers)
        
    n_jets_per_event = np.histogram( df.entry.values ,
                                        bins=np.linspace(-0.5+min_event,min_event+n_events+0.5-1,n_events+1))[0]
    
    total_jets = len(df)
    #print('total ',total_jets)
    splitindices = []
    running_sum = 0
    event_sum = 0
    combinations = []
    jet_n_neighbors = []
    for n_jets in n_jets_per_event:
        
        for _ in range(n_jets):
            jet_n_neighbors.append((n_jets-1))
            splitindices.append( (n_jets-1)+running_sum)
            running_sum+=(n_jets-1)
        comb = np.array(list(itertools.product(np.arange(n_jets), repeat=2)))

        combinations.append( comb+event_sum )
        
        event_sum+=n_jets
    
    
    combinations = np.concatenate(combinations)

    eta1 = df.jet_eta.values[combinations[:,0]]
    phi1 = df.jet_phi.values[combinations[:,0]]
    
    eta2 = df.jet_eta.values[combinations[:,1]]
    phi2 = df.jet_phi.values[combinations[:,1]]
    
    jet2_flavor = df.jet_label.values[combinations[:,1]]
    
    dRs = deltaR(eta1,phi1,eta2,phi2)
    jet2_flavor = jet2_flavor[dRs > 0]
    dRs = dRs[dRs > 0]
    dRs = np.split(dRs,splitindices)
    jet2_flavor = np.split(jet2_flavor,splitindices)
    
    
    
    return dRs[:-1],jet2_flavor[:-1],jet_n_neighbors
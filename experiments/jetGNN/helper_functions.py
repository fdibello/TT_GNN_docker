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

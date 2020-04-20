import uproot
import numpy as np
import pandas as pd
import sys

f = uproot.open(sys.argv[1]) #'../dataset10M_sameFlav.root'

n_entries = f['t1'].numentries


def collect_jets(entrystart,entry_stop):
    df = f['t1'].pandas.df(['jet_label'],
                           entrystart=entrystart,entrystop=entry_stop).reset_index()
    return df

jets_test_data = collect_jets(0,n_entries)


old_pred = np.load(sys.argv[2])

where_light = jets_test_data['jet_label'] == 3

# nn = old_pred[where_light]

# light_jet_factor = 1

# old_pred[where_light] = 1.0/(((1/nn)-1)*(1/light_jet_factor)+1)

np.save('adjusted_prediction_'+sys.argv[3]+'.npy',np.mean(old_pred,axis=1) )
np.save('single_prediction_'+sys.argv[3]+'.npy',old_pred[:,0] )
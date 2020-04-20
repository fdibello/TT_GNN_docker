


import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm


f = uproot.open('datasetB_6jets.root')



n_entries = f['t1'].numentries
training_split = 600000




def collect_jets(entrystart,entry_stop):
    jets_data = []
    jets_target = []
    for jet_i in range(6):
        if jet_i == 0:
            suffix = ''
        else:
            suffix = str(jet_i+1)
        jet_array = f['t1'].pandas.df(['pT'+suffix,'eta'+suffix,'phi'+suffix,'label'+str(jet_i+1)],
                                                  entrystart=entrystart,entrystop=entry_stop).values
        target_array = f['t1'].pandas.df(['isTag'+suffix],entrystart=entrystart,entrystop=entry_stop).values

        jet_index = np.tile(jet_i+1,len(jet_array))
        
        event_index = np.arange(entrystart,entry_stop)
        
        jet_array = np.column_stack( (event_index,jet_index,jet_array ) )
        
        jets_data.append( jet_array )
        jets_target.append( target_array )
    jets_data = np.concatenate(jets_data)
    jets_target = np.concatenate(jets_target)
    
    # pt > 0 means its a jets, not a placeholder
    valid_jets = np.where(jets_data[:,2] > 0)[0]
    jets_data = jets_data[valid_jets]
    jets_target = jets_target[valid_jets]
    
    columns=['eventIdx','jetIdx','pt','eta','phi','label']
    df = pd.DataFrame(columns=columns+['istagged'])
    c_dtypes = [np.int32,np.int32,np.float64,np.float64,np.float64,np.int32]
    for column_i, (column,c_dtype) in enumerate(zip(columns,c_dtypes)):
        df[column] = jets_data[:,column_i].astype(c_dtype)
    df['istagged'] = jets_target.astype(np.int32)
    
    df = df.sort_values('eventIdx').reset_index(drop=True)
    return df

jets_train_data = collect_jets(0,training_split)




var_transformations = {}
for var_i,var  in enumerate(['pt','eta','phi','label']):
    var_values = jets_train_data[var].values
    var_transformations[var] = [np.mean(var_values),np.std(var_values)]



import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import math



class JetEventsDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()
        
        self.event_numbers = list(set(self.df['eventIdx'].values))
        self.n_events = len(self.event_numbers)
        
        min_event = np.amin(self.event_numbers)
        
        self.event_njets = np.histogram( self.df.eventIdx.values ,
                                        bins=np.linspace(-0.5+min_event,min_event+self.n_events+0.5-1,self.n_events+1))[0]
        
        for col in ['pt','eta','phi','label']:
            mean, std = var_transformations[col]
            self.df[col] = (self.df[col].values-mean)/std
        
        all_xs = self.df[['pt','eta','phi','label']].values
        
        splitindices = []
        running_sum = 0
        for idx in self.event_njets:
            splitindices.append(idx+running_sum)
            running_sum+=idx
        
        self.xs = np.split(all_xs,splitindices)

        self.ys = np.split(self.df.istagged.values,splitindices)
        
        
    def __len__(self):
       
        return self.n_events


    def __getitem__(self, idx):
        
        x,y = torch.FloatTensor(self.xs[idx]),torch.FloatTensor(self.ys[idx])
        
        return x, y


class JetEventsSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the events
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]





ds = JetEventsDataset(jets_train_data)


batch_size = 250
batch_sampler = JetEventsSampler(ds.event_njets, batch_size)
data_loader = DataLoader(ds, batch_sampler=batch_sampler)



class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        self.query = nn.Sequential(
            nn.Linear(in_features, small_in_features),
            nn.Tanh(),
        )
        self.key = nn.Linear(in_features, small_in_features)

    def forward(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)  # B,N,C/10

        x = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # B,N,N

        x = x.transpose(1, 2)  # (B,N,N)
        x = x.softmax(dim=2)  # over rows
        x = torch.matmul(x, inp)  # (B, N, C)
        return x

class DeepSet(nn.Module):
    def __init__(self, in_features, feats):
        """
        DeepSets implementation
        :param in_features: input's number of features
        :param feats: list of features for each deepsets layer
       
        """
        super(DeepSet, self).__init__()

        layers = []

        layers.append(DeepSetLayer(in_features, feats[0]))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(DeepSetLayer(feats[i-1], feats[i]))

        self.sequential = nn.Sequential(*layers)

        self.node_classifier = nn.Conv1d(feats[-1], 1, 1)
        
    def forward(self, x):
        
        
        x = self.sequential(x)
        
        x = self.node_classifier(x.transpose(2,1)).transpose(1,2)
        
        return x


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        DeepSets single layer
        :param in_features: input's number of features
        :param out_features: output's number of features
        """
        super(DeepSetLayer, self).__init__()

        self.attention = Attention(in_features)
        self.layer1 = nn.Conv1d(in_features, out_features, 1)
        self.layer2 = nn.Conv1d(in_features, out_features, 1, bias=True)

    def forward(self, x):
        # x.shape = (B,N,C)
        x_T = x.transpose(2, 1)  # B,N,C -> B,C,N
        
        x = self.layer1(x_T) + self.layer2(self.attention(x).transpose(1, 2))
        
        # normalization
        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
        
        x = x.transpose(2, 1) # B,N,C -> B,C,N
        return x




net = DeepSet(4,[10,10])


loss_vs_epoch = []

lossfunc = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


epochs = 550

net.cuda()
net.train();

for epoch in range(epochs):
    
    batch_losses = []
    for x,y in tqdm(data_loader):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        
        output = net(x)
        
        y = y.double()
        y = y.unsqueeze(-1)
        
        loss = lossfunc(output,y)
        batch_losses.append(loss.item())
        
        loss.backward()  
        optimizer.step()
    
    if np.mean(batch_losses) < np.amin(np.array(loss_vs_epoch)[:,1]):
        torch.save(net.state_dict(), 'model.pt')


    loss_vs_epoch.append([epoch,np.mean(batch_losses)])

    print(loss_vs_epoch[-1])


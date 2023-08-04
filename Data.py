import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

normalize=True

class MY_Dataset(Dataset):
    def __init__(self, features, targets):
        super().__init__()
        self.features = torch.from_numpy(features).type(torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


raw_data=np.loadtxt("nucl.txt")
num_of_nucleus=int(len(raw_data)/3)
ZA=np.zeros((num_of_nucleus,2))
delta_E=np.zeros((num_of_nucleus,1))
for k in range(num_of_nucleus):
    ZA[k][0]=raw_data[3*k]
    ZA[k][1]=raw_data[3*k+1]
    delta_E[k][0]=raw_data[3*k+2]    

if normalize:
    delta_E-=np.mean(delta_E)
    delta_E/=(np.var(delta_E))**0.5

DS=MY_Dataset(ZA,delta_E)


train_part=0.5
train_length=int(train_part*num_of_nucleus)
test_length=num_of_nucleus-train_length
train, test = random_split(dataset=DS, lengths=[train_length, test_length])


train_loader = DataLoader(train, batch_size=len(train), shuffle=True)
test_loader = DataLoader(test, batch_size=len(test), shuffle=True)
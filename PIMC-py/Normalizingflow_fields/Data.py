import torch
from torch.utils.data import Dataset


class Distribution_set(Dataset):
    def __init__(self, distribution,epoch_size=2**14):
        super().__init__()
        self.distribution = distribution
        self.features = torch.tensor(0)
        self.n_sample = epoch_size

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.distribution.sample((1,))[0]
    
    def sample(self,n_sample):
        self.features = self.distribution.sample((n_sample,))
        



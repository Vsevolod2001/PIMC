import torch 
from torch import nn
import numpy as np
pi = torch.tensor(np.pi)


class KL_with_S(nn.Module):
    
    def __init__(self,S,lattice):
        super().__init__()
        self.S = S
        self.lattice = lattice
        self.val_S = 0
        self.normalizer = (self.lattice.total_nodes / 2) * (1 + torch.log( 2 * pi ))
    
    def forward(self,x,log_abs_det):
        self.val_S = self.S(x)
        loss = torch.mean(self.val_S-log_abs_det)
        loss -= self.normalizer
        return loss
        
    def ESS(self,latent_log_prob,log_abs_det):
        with torch.no_grad():
            Lnlike = self.val_S-log_abs_det+latent_log_prob
            Lnlike-=torch.mean(Lnlike)
            if torch.mean(torch.abs(Lnlike))>10:
                return 0
            like = torch.exp(Lnlike)
            ess = (torch.mean(like) ** 2)/(torch.mean(like**2))
        return ess


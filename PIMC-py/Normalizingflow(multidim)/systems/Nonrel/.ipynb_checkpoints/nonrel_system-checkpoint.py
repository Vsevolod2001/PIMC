from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class nonrel_System(System):
    
    def __init__(self,n_nod,beta,dim=1):
        super().__init__(n_nod,beta,dim)
        self.normalizer = self.dim * 0.5 * torch.log( 2 * pi * self.a ) 
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return torch.sum(t , dim=-1)
    
    def F_T(self,diff):
        return diff / (self.a ** 2)   
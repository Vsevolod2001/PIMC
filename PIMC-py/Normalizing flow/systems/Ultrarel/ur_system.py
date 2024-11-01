from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class ur_System(System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
        self.normalizer =  torch.log( pi * self.a ) 
    
    def T(self,diff):
        a=self.a
        t=( 1 / a ) * torch.log(1 + (diff / a) ** 2)
        return t
    
    def Y(self,diff):
        v=diff / self.a
        return 2 / (self.a ** 2) * v / (1 + v ** 2)

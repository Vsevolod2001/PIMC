import torch
from LOSS import KL_with_S
from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Pot(System):
    
    def __init__(self,beta,dim=1):
        super().__init__(1,beta,dim)
        self.normalizer = self.dim * 0.5 * torch.log( 2 * pi * self.a ) 
    
    def T(self,diff):
        return torch.sum(0*diff,-1)

    
    def F_T(self,diff):
        return 0*diff   
        
        
    
    
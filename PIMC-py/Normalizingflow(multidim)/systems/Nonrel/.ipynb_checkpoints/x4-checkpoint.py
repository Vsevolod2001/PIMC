from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class X4(nonrel_System):
    
    def __init__(self,n_nod,beta,dim,**args):
        super().__init__(n_nod,beta,dim)
        self.g = args["g"]
        
    def V(self,x):
        return torch.sum(x ** 2 / 2,dim=-1) + self.g * (torch.sum(x ** 2,dim=-1)) ** 2
        
    def U(self,x):
        return -x - 4 * self.g * (torch.sum(x ** 2,dim=-1)) * x
        
    

      
        
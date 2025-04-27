import numpy as np
import torch
from systems.Pot.pot import Pot
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Power(Pot):
    
    def __init__(self,beta,dim=1,**args):
        super().__init__(beta,dim)
        self.normalizer = 0
        self.p = args["p"]
        self.g = args["g"]
    
        
    def V(self,x):
        return self.g * (torch.sum(x**2,2)) ** (self.p//2)
    
    def F_V(self,x):
        fv = torch.zeros(x.size(0),1,self.dim)
        for _ in range(self.dim):
            fv[:,:,_] = -2 * (self.p//2) * self.g * x[:,:,_] * (torch.sum(x ** 2,dim=-1)) ** (self.p//2-1)
        return fv
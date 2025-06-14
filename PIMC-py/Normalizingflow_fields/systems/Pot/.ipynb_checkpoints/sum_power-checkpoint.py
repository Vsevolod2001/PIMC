import numpy as np
import torch
from systems.Pot.pot import Pot
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Sum_power(Pot):
    
    def __init__(self,beta,dim=1,**args):
        super().__init__(beta,dim)
        self.normalizer = 0
        self.p = args["p"]
        self.g = args["g"]
    
        
    def V(self,x):
        return 0.5 * torch.sum(x**2,2) + self.g * torch.sum(x ** self.p,2)
    
    def F_V(self,x):
        fv = -x - self.g * self.p * x ** (self.p-1)  
        return fv
import numpy as np
import torch
from systems.Pot.pot import Pot
from LOSS import KL_with_S

pi=torch.tensor(np.pi)

class Circle(Pot):
    
    def __init__(self,beta,dim=1,**args):
        super().__init__(beta,dim)
        self.normalizer = 0
        self.g = args["g"]
        self.x0 = args["x0"]
    
        
    def V(self,x):
        return self.g * (torch.sum(x**2,2) - self.x0 ** 2 ) ** 2
    
    def F_V(self,x):
        fv = torch.zeros(x.size(0),1,self.dim)
        for _ in range(self.dim):
            fv[:,:,_] = -4 * self.g * x[:,:,_] * (torch.sum(x ** 2,dim=-1) - self.x0 ** 2)
        return fv

      
        
from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch
pi = torch.tensor(np.pi)

class Columb(nonrel_System):
    
    def __init__(self,n_nod,beta,dim,**args):
        super().__init__(n_nod,beta,dim)
        self.alpha=args["alpha"]
        self.R=args["R"]
    
    def V(self,x):
        return  -self.alpha * (torch.sum(x ** 2,dim = -1) + self.R ** 2) ** (-0.5)
    
    def F_V(self,x):
        return -self.alpha * x * (torch.sum(x ** 2,dim=-1) + self.R ** 2) ** (-1.5)
    



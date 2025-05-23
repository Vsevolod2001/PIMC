from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Two_wells(nonrel_System):
    
    def __init__(self,n_nod,beta,dim,**args):
        super().__init__(n_nod,beta,dim)
        self.g = args["g"]
        self.x0 = args["x0"]
        
    def V(self,x):
        return self.g * (torch.sum(x ** 2,dim=-1) - self.x0 ** 2 ) ** 2
        
    def F_V(self,x):
        fv = torch.zeros(x.size(0),self.n_nod,self.dim)
        for _ in range(self.dim):
            fv[:,:,_]=-4 * self.g * x[:,:,_] * (torch.sum(x ** 2,dim=-1) - self.x0 ** 2)
        return fv
    

      
        
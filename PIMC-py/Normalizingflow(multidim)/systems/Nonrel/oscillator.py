from systems.Nonrel.nonrel_system import nonrel_System
import numpy as np
import torch
from systems.Nonrel.osc_andist import calc_Z, get_A
pi=torch.tensor(np.pi)

class Oscillator(nonrel_System):
    
    def __init__(self,n_nod,beta,dim):
        super().__init__(n_nod,beta,dim)
        self.Log_Z = self.dim * torch.log(torch.tensor(calc_Z(n_nod,beta)))
        
    def V(self,x):
        return torch.sum(x ** 2 / 2, dim=-1)
    
    
    def U(self,x):
        return -x
      
    def theor_Psi2(self,z):
        return (pi) ** (-0.5) * np.exp(-z ** 2)     
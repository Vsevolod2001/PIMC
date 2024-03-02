from System import system
from NFconstants import a
import numpy as np
import torch
from torch.special import scaled_modified_bessel_k1 as Q1

class rel_oscillator(system):
    
    def __init__(self,sigma,hbar=1,m=1,w=1):
        super().__init__(m, hbar)
        self.w=w
        self.sigma=sigma
    
    def T(self,diff):
        y=( 1 + (diff/a)**2 * self.sigma)**0.5
        ln=torch.log(Q1( a*y / self.sigma )/y)
        t = (y-1) / self.sigma - ln / a
        
        
        pi=torch.tensor(np.pi)
        delta=0.5 * torch.log( 2*a / (pi * self.sigma)) / a
        return t-delta
    
    def V(self,x):
        return self.m * (self.w) ** 2 * x ** 2 / 2
    


nr_rel_oscillator=rel_oscillator(0.0001)
ur_rel_oscillator=rel_oscillator(10000)

rel_oscillator_100=rel_oscillator(100)

unit_rel=rel_oscillator(1)
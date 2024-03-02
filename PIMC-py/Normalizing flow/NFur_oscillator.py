from System import system
from NFconstants import a
import numpy as np
import torch

class ur_oscillator(system):
    
    def __init__(self,sigma,hbar=1,m=1,w=1):
        super().__init__(m, hbar)
        self.w=w
        self.sigma=sigma
    
    def T(self,diff):
        t=( 1 / a ) * torch.log(1 + (diff / a) ** 2)
        
        pi=torch.tensor(np.pi)
        delta=0.5* torch.log( 2 * self.sigma / (pi* a) ) / a + 1/self.sigma
        return t-delta
    
    def V(self,x):
        return self.m * (self.w) ** 2 * x ** 2 / 2


basic_ur=ur_oscillator(1)
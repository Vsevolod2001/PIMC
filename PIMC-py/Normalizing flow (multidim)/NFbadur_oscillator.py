from System import System
from NFconstants import a
import numpy as np
import torch

pi=torch.tensor(np.pi)

class BadUr_Oscillator(System):
    
    def __init__(self,n_nod,beta):
        super().__init__(n_nod,beta)
        self.normalizer =  torch.log( pi * self.a ) 
    
    def T(self,diff):
        a=self.a
        t=( 1 / a ) * (diff / a) ** 2
        return t
    
    def V(self,x):
        return x ** 2 / 2

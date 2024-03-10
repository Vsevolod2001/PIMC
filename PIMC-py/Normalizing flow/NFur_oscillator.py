from System import System
from NFconstants import a
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Ur_Oscillator(System):
    
    def __init__(self):
        super().__init__()
        self.normalizer =  torch.log( pi * a ) 
    
    def T(self,diff):
        t=( 1 / a ) * torch.log(1 + (diff / a) ** 2)
        return t
    
    def V(self,x):
        return x ** 2 / 2

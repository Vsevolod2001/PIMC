from System import System
from NFconstants import a
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Oscillator(System):
    
    def __init__(self):
        super().__init__()
        self.normalizer = 0.5 * torch.log( 2 * pi * a )
    
    def T(self,diff):
        t = (diff) ** 2 / (2 * a ** 2)
        return t
        
    def V(self,x):
        return x ** 2 / 2
    
        
from System import System
from NFconstants import a
import numpy as np
import torch
from torch.special import scaled_modified_bessel_k1 as Q1
from dimensionlesser import get_coeffs
pi=torch.tensor(np.pi)
a=torch.tensor(a)

class Rel_Oscillator(System):
    
    def __init__(self,**args):
        super().__init__()
        self.sigma=args["sigma"]
        self.s1, self.s2, self.s3 = get_coeffs(self.sigma)
        self.normalizer = torch.log(pi * self.s1 / (self.s2 * Q1(self.s2 * a))) 
    
    def T(self,diff):
        y=( 1 + (diff/a)**2 * (self.s1) ** (-2))**0.5
        ln=torch.log(Q1( self.s2 * a * y )/(y * Q1(self.s2 * a)))
        t = self.s2 * (y-1) - ln / a
        return t
    
    def V(self,x):
        return  self.s3 * x ** 2 / 2
    



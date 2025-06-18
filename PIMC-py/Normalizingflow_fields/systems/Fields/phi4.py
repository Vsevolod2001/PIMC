from systems.Fields.scalar_field import Scalar_Field
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Phi4(Scalar_Field):

    def __init__(self,lattice,mass2,g):
        super().__init__(lattice,mass2)
        self.normalizer = self.dim * 0.5 * torch.log( 2 * pi * self.a ) 
        self.g = g
         

    def V_int(self,phi):
        return self.g * torch.sum(phi ** 4, dim=2)

    def F_int(self,phi):
        return -4 * self.g * phi ** 3 
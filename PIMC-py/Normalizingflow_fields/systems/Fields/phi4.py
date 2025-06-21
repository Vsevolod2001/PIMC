from systems.Fields.scalar_field import Scalar_Field
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Phi4(Scalar_Field):

    def __init__(self,lattice,mass2,g):
        super().__init__(lattice,mass2) 
        self.g = g
         

    def V_int(self,phi):
        return self.g * torch.sum(phi ** 4, dim = -1)

    def F_int(self,phi):
        return -4 * self.g * phi ** 3 
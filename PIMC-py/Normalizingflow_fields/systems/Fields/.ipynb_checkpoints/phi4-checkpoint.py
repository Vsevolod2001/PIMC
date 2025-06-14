from systems.Fields.scalar_field import Scalar_Field
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Phi4(Scalar_Field):

    def __init__(self,n_nod,beta,space_dim,mass2,L,R,g):
        super().__init__(n_nod,beta,space_dim,mass2,L,R)
        self.normalizer = self.dim * 0.5 * torch.log( 2 * pi * self.a ) 
        self.h = R/L
        self.L = L
        self.R = R
        self.mass2 = mass2
        self.space_dim = space_dim
        self.g = g
         

    def V_int(self,phi):
        return self.g * torch.sum(phi ** 4, dim=2)

    def F_int(self,phi):
        return -4 * self.g * phi ** 3 
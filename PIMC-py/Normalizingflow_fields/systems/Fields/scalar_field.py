from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Scalar_Field(System):

    def __init__(self,lattice,mass2):
        super().__init__(lattice)
        self.normalizer = self.lattice.n_dims * 0.5 * torch.log( 2 * pi * self.lattice.steps[0]) 
        self.mass2 = mass2
        self.J = torch.zeros((self.L,self.L))
        


    def Kin(self,phi):
        
        return 0.5 * torch.einsum("bi,ij,bj->b",phi,self.lattice.kin_mat,phi)


    def V(self,phi):
        
        return self.V_mass(phi) + self.V_J(phi) + self.V_int(phi)

   

    def V_mass(self,phi):
        
        return torch.sum(self.mass2 * phi ** 2 / 2,-1)

    def V_J(self,phi):
        
        return torch.matmul(phi,self.J)

    def V_int(self,phi):
        return 0

    
    
    def set_J_global(self,j):
        self.J = j * torch.ones(self.L)

    def set_J_local(self,j,k_nod):
        self.J[k_nod] = j

    def set_J(self,J):
        self.J = J.clone().detach()
    
    def S(self,x):
        s = (self.Kin(phi)+self.V(phi)) * self.lattice.vol_element
        return s 

    
    
    
    
    
    def F_kin(self,phi):
        
        return 0

    def F_mass(self,phi):
        return -self.mass2 * phi

    def F_int(self,phi):
        return 0

    def F_J(self,phi):
        return -self.J
    

    def F_V(self,phi):
        return self.F_grad(phi) + self.F_mass(phi) + self.F_int(phi) + self.F_J(phi)

    def F(self,phi):
        return (self.h ** self.space_dim) * self.a * (self.F_T(phi)+self.F_V(phi))

    

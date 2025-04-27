from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Scalar_Field(System):

    def __init__(self,n_nod,beta,space_dim,mass2,L,R):
        super().__init__(n_nod,beta,L ** space_dim)
        self.normalizer = self.dim * 0.5 * torch.log( 2 * pi * self.a ) 
        self.h = R/L
        self.L = L
        self.R = R
        self.mass2 = mass2
        self.space_dim = space_dim
        self.J = torch.zeros(self.L)
        
    def T(self,diff):
        t = (diff) ** 2 / (2 * self.a ** 2)
        return torch.sum(t , dim=2)

    def V(self,phi):
        return self.V_grad(phi) + self.V_mass(phi) + self.V_J(phi) + self.V_int(phi)

    def V_grad(self,phi):
        
        v_grad = 0
        
        for k in range(self.space_dim):
            grad = torch.roll(phi,self.L**k,2) - phi
            v_grad += torch.sum(grad**2,2)
        
        return v_grad / (2 * self.h ** 2)    

    def V_mass(self,phi):
        return torch.sum(self.mass2 * phi ** 2 / 2,2)

    def V_J(self,phi):
        
        v_J = 0
        
        for i in range(self.L):
            v_J += self.J[i] * phi[:,:,i]
        
        return v_J

    def V_int(self,phi):
        return 0

    def set_J_global(self,j):
        self.J = j * torch.ones(self.L)

    def set_J_local(self,j,k_nod):
        self.J[k_nod] = j

    def set_J(self,J):
        self.J = J.clone().detach()
    
    def Full_S(self,x):
        full_S = 0
        x_next = torch.roll(x,-1,1)
        diff = x_next-x
        full_T = torch.sum(self.T(diff),dim=1)
        full_V = torch.sum(self.V(x),dim=1)
        full_S = (self.h ** self.space_dim) * self.a * (full_T + full_V) + self.n_nod * self.normalizer + self.Log_Z
        return full_S 

    def F_grad(self,phi):
        f_grad = 0
        
        for k in range(self.space_dim):
            lapl = torch.roll(phi,self.L**k,2) + torch.roll(phi,-self.L**k,2) - 2 * phi
            f_grad += lapl
        
        return f_grad / self.h ** 2

    def F_mass(self,phi):
        return -self.mass2 * phi

    def F_int(self,phi):
        return 0

    def F_J(self,phi):
        return -self.J
    
    def F_T(self,phi): #=-dT/d phi
        f_T = 0
        d2 = torch.roll(phi,1,1) + torch.roll(phi,-1,1) - 2 * phi
        f_T += d2
        return f_T / self.a ** 2

    def F_V(self,phi):
        return self.F_grad(phi) + self.F_mass(phi) + self.F_int(phi) + self.F_J(phi)

    def F(self,phi):
        return (self.h ** self.space_dim) * self.a * (self.F_T(phi)+self.F_V(phi))

    

from systems.System import System
import numpy as np
import torch

pi=torch.tensor(np.pi)

class Scalar_Field(System):

    def __init__(self,lattice,mass2):
        super().__init__(lattice)
        self.mass2 = mass2
        self.J = torch.zeros((self.lattice.total_nodes)).to(lattice.device)

        
        self.log_Z = self.get_log_Z()
        


    def Kin(self,phi):
        return 0.5 * torch.einsum("bi,ij,bj->b",phi,self.lattice.kin_mat,phi)


    def V(self,phi):
        
        return self.V_mass(phi) + self.V_J(phi) + self.V_int(phi)

   

    def V_mass(self,phi):
        
        return 0.5 * torch.sum(self.mass2 * phi ** 2 ,-1)

    def V_J(self,phi):
        
        return torch.einsum("bj,j->b",phi,self.J)

    def V_int(self,phi):
        return 0

    
    
    def set_J_global(self,j):
        self.J = j * torch.ones(self.lattice.total_nodes).to(self.lattice.device)
        self.log_Z += self.get_log_Z_J()

    def set_J_local(self,j,multi_index):
        for i in range(self.lattice.n_nodes[0]):
            multi_index[0] = i
            index = self.lattice.multi_to_index(multi_index)
            self.J[index] = j
        self.log_Z += self.get_log_Z_J()    

    def set_J(self,J):
        self.J = J.clone().detach()
        self.log_Z += self.get_log_Z_J()
    
    def S(self,phi):
        s = (self.Kin(phi)+self.V(phi)) * self.lattice.vol_element + self.log_Z
        return s 

    
    
    
    
    
    def F_kin(self,phi):
        return -torch.einsum("ij,bj->bi",self.lattice.kin_mat,phi)

    def F_mass(self,phi):
        return -self.mass2 * phi

    def F_int(self,phi):
        return 0

    def F_J(self,phi):
        return -self.J
    

    def F_V(self,phi):
        return  self.F_mass(phi) + self.F_int(phi) + self.F_J(phi)

    def F(self,phi):
        return self.lattice.vol_element * (self.F_kin(phi)+self.F_V(phi))
    
    
    def get_free_prop_p(self):

        return (self.lattice.get_diag_kin_mat() + self.mass2) ** -1
   

    def get_free_prop_x(self):

        prop = torch.diag(self.get_free_prop_p())
        prop = torch.einsum("ki,ij,mj->km",self.lattice.ort_mat,prop,self.lattice.ort_mat)
        return prop

    def get_log_Z(self):
        if self.mass2>0:
            vK = self.lattice.vol_element * (self.lattice.get_diag_kin_mat() + self.mass2)
            vK = vK/(2*pi)
            return -0.5 * torch.sum(torch.log(vK))
        else:
            return 0
    
    def get_log_Z_J(self):
        if self.mass2>0:
            J_p = torch.matmul(self.lattice.ort_mat_t,self.J)
            prop_p = self.get_free_prop_p()
            return self.lattice.vol_element * 0.5 * torch.dot(prop_p * J_p,J_p)
        else:
            return 0


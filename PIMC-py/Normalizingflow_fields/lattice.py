import numpy as np
import torch
from transforms import t_get_O

class Lattice:
    def __init__(self,n_nodes,sizes):
        self.n_dims = len(n_nodes)
        self.n_nodes = np.array(n_nodes)
        self.sizes = np.array(sizes)
        self.steps =  self.n_nodes/self.sizes
        self.total_nodes = np.prod(self.n_nodes)
        
        
        self.ort_mat = t_get_O(self.n_nodes[0])
        for dir in range(1,self.n_dims):
            self.ort_mat = torch.kron(self.ort_mat,t_get_O(self.n_nodes[dir]))
        self.ort_mat_t = torch.t(self.ort_mat)    

    def get_pair_split_masks_field(self,dir):
        bigmask = list(range(self.total_nodes))
        mask1 = list(filter(lambda x: x% (4 * self.n_nodes[dir] ** dir) < 2 * self.n_nodes[dir] ** dir,bigmask))       
        mask2 = list(filter(lambda x: x%4>=2,bigmask))
        return [mask1, mask2]
    

    def get_shift_mat(size):
        id = torch.eye(size)
        T = torch.roll(id,1,-1)
        return T
    
    def get_k_mu(self,dir):
        id = torch.eye(self.n_nodes[dir])

        left = id.clone().detach()
        for _ in range(dir-1):
            left = torch.kron(left,id)


        shift = Lattice.get_shift_mat(self.n_nodes[dir])
        shift_dagger = torch.t(shift)
        m = 2 * id - shift - shift_dagger
       

        right = id.clone().detach()
        for _ in range(self.n_dims-dir-2):
            right = torch.kron(right,id)
    
        
        k_mu = m
        
        if dir>0:
            k_mu = torch.kron(left,k_mu)

        if dir < self.n_dims - 1:    
            k_mu = torch.kron(k_mu,right)

        return k_mu    

    def kin_mat(self):
        kin_mat = torch.zeros((self.total_nodes,self.total_nodes))
        for dir in range(self.n_dims):
            kin_mat += self.get_k_mu(dir)/self.steps[dir] ** 2
        return kin_mat    




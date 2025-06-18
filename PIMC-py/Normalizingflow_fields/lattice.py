import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Data import Distribution_set

class Lattice:
    
    
    def __init__(self,n_nodes,sizes):
        self.n_dims = len(n_nodes)
        self.n_nodes = np.array(n_nodes)
        self.sizes = np.array(sizes)
        self.steps =  self.n_nodes/self.sizes
        self.total_nodes = np.prod(self.n_nodes)
        self.vol_element = np.prod(self.steps)
        
        self.dtype = torch.float64
        self.device = "cuda"
        
        self.ort_mat = self.get_big_ort_mat().to(self.device)
        self.ort_mat_t = torch.t(self.ort_mat)

        self.kin_mat = self.get_kin_mat().to(self.device)   

    
    
    
    def get_pair_split_masks_field(self,dir):
        bigmask = list(range(self.total_nodes))
        mask1 = list(filter(lambda x: x% (4 * self.n_nodes[dir] ** dir) < 2 * self.n_nodes[dir] ** dir,bigmask))       
        mask2 = list(filter(lambda x: x%4>=2,bigmask))
        return [mask1, mask2]
    

    def get_ort_mat(N,dtype=torch.float64,fmt="torch"):

        n_cols = (N+1)//2-1

        column = torch.arange(1,n_cols+1,dtype=torch.float64)
        line = torch.arange(N,dtype=dtype)
        args = (2*np.pi/N) * torch.einsum('i,j->ij',line,column)

        ort = torch.zeros((N,2*n_cols),dtype=dtype)
        ort[:,np.arange(0,2*n_cols,2)] = torch.cos(args)
        ort[:,np.arange(1,2*n_cols,2)] = torch.sin(args)
        ort = (2/N)**0.5 * ort

        first = torch.ones((N,1),dtype=dtype)
        if N%2==0:
            signed = torch.ones((N,1))
            signed[np.arange(1,N,2),:] = -1
            first = torch.cat((first,signed),1)
        first = first * N ** (-0.5)
    
        ort = torch.cat((first,ort),1)

        if fmt == "np":
            ort = ort.numpy()

        return ort    
    
    
    def get_big_ort_mat(self):
        ort_mat = Lattice.get_ort_mat(self.n_nodes[0])
        for dir in range(1,self.n_dims):
            ort_mat = torch.kron(ort_mat,Lattice.get_ort_mat(self.n_nodes[dir]))
        return ort_mat     



    def get_shift_mat(size):
        id = torch.eye(size)
        T = torch.roll(id,1,-1)
        return T
       

    
    def id_wap(self,mat,pos):    

        wap = mat.clone().detach()
        
        for i in range(pos-1,-1,-1):
            wap = torch.kron(torch.eye(self.n_nodes[i],dtype=self.dtype),wap)    
        
        for i in range(pos+1,self.n_dims):       
            wap = torch.kron(wap,torch.eye(self.n_nodes[i],dtype=self.dtype))    
        return wap    


    def get_deriv_mat(self,dir):
        id = torch.eye(self.n_nodes[dir],dtype=self.dtype)
        shift = Lattice.get_shift_mat(self.n_nodes[dir]).to(self.dtype)
        shift_dag = torch.t(shift)
        m = 2 * id - shift - shift_dag
        m = m / self.steps[dir] ** 2
        
        return m


    def get_kin_mat(self): 
        
        kin_mat = torch.zeros((self.total_nodes,self.total_nodes),dtype=self.dtype)
        
        for dir in range(self.n_dims):
            
            kin_mat += self.id_wap(self.get_deriv_mat(dir),dir)
        
        return kin_mat    

    def get_diag_deriv(self,dir):
        
        n_twodim = self.n_nodes[dir]-1-int(self.n_nodes[dir]%2==0)
        args = torch.arange(0,n_twodim,dtype=int)
        args = (args//2+1).to(self.dtype)
        args = np.pi * args / self.n_nodes[dir]

        diag = 4 * (torch.sin(args) ** 2)
        if self.n_nodes[dir]%2 == 0:
            first = torch.tensor([0,4],dtype=self.dtype)
        else:
            first = torch.tensor([1],dtype=self.dtype)
        diag = torch.cat((first,diag))
        
        diag = diag / self.steps[dir] ** 2 
        return diag        
    
    def get_diag_kin_mat(self):
        diag_kin = torch.zeros((self.total_nodes,self.total_nodes),dtype = self.dtype)
        for dir in range(self.n_dims):
            diag_mat = torch.diag(self.get_diag_deriv(dir))
            diag_kin += self.id_wap(diag_mat,dir)
        return torch.diagonal(diag_kin).to(self.device)    




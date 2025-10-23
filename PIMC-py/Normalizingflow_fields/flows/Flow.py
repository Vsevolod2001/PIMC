import torch
from torch import nn
from typing import Callable, List, Tuple
from lattice import Lattice


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flow(nn.Module):
    
    def __init__(self,lattice,ort):
        super().__init__()
        self.ort = ort
        self.lattice = lattice


    
    def save(self,filename):
        
        pass
    
    def load_model(filename):   
        
        pass


    

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        pass
    
    
    def f(self, x: torch.Tensor,params=torch.tensor([])) -> Tuple[torch.Tensor, torch.Tensor]:
        
        pass

    def g_f_test(self,z):
        
        z0 = torch.clone(z)
        x, log_abs_det_g = self.g(z)
        y, log_abs_det_f = self.f(x)
        return torch.norm(z0-y), torch.norm(log_abs_det_g+ log_abs_det_f)


        

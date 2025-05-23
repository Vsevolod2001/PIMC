import torch
from torch import nn
from typing import Callable, List, Tuple


from flows.theta import ThetaNetwork
from flows.Layers import AffineCouplingLayer
from transforms import get_split_masks
from transforms import get_pair_split_masks

class NormalizingFlow(nn.Module):
    
    def __init__(self, flows: List[nn.Module], O=torch.tensor([])):
        super().__init__()
        self.flows = flows
        self.O = O
        self.Ot = torch.t(O)
        self.ort = self.O.shape[0]>0
        
    def configure_flows(n_flows,num_hidden,hidden_dim,p_drop,dim,param_dim=0,mask_config = get_pair_split_masks,sys_dim = 1):
        flows = []
        split_masks_d = mask_config(dim)
    
        for k in range(n_flows):
            theta = ThetaNetwork.configure_theta(num_hidden = num_hidden, 
                                                 hidden_dim = hidden_dim, 
                                                 p_drop = p_drop,
                                                 in_dim = (dim//2 + param_dim) * sys_dim ,
                                                 out_dim = dim//2 * sys_dim)
            flows.append(AffineCouplingLayer(theta, split = split_masks_d, swap = k % 2))
   
        flows = nn.ModuleList(flows)
        return flows     

    

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        x, sum_log_abs_det = z, torch.zeros(z.size(0)).to(z.device)
        for flow in reversed(self.flows):
            x, log_abs_det = flow.g(x,params)
            sum_log_abs_det += log_abs_det
        
        if self.ort:
            x = torch.matmul(x,self.Ot.to(x.device))
            
        return x, sum_log_abs_det
    
        
    def __len__(self) -> int:
        return len(self.flows)
    
    def f(self, x: torch.Tensor,params=torch.tensor([])) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            res = x.clone()
            if self.ort:
                res = torch.matmul(res,self.O.to(res.device))
        
            z, sum_log_abs_det = res, torch.zeros(res.size(0)).to(res.device)
        
            for flow in self.flows:
                z, log_abs_det = flow.f(z,params)
                sum_log_abs_det += log_abs_det
        
        return z, sum_log_abs_det

    def append_aff(self,hidden_dim,num_hidden,num_aff):
        last = self.flows[-1]
        split = last.split
        swap = last.swap
        in_dim = last.theta.in_dim
        out_dim = last.theta.out_dim
        for i in range(num_aff):
            theta = ThetaNetwork.configure_theta(num_hidden = num_hidden, 
                                                 hidden_dim = hidden_dim, 
                                                 p_drop = 0.0,
                                                 in_dim = in_dim,
                                                 out_dim = out_dim)
            self.flows.append(AffineCouplingLayer(theta, split = split, swap = (swap+i+1)%2))
        
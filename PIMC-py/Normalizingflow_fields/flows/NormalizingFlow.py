import torch
from torch import nn
from typing import Callable, List, Tuple

from lattice import Lattice

from flows.theta import ThetaNetwork
from flows.Layers import AffineCouplingLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizingFlow(nn.Module):
    
    def __init__(self, flows: List[nn.Module],lattice,ort):
        super().__init__()
        self.flows = flows
        self.ort = ort
        self.lattice = lattice

        self.n_flows = len(flows)
        self.num_hidden = self.flows[0].theta.num_hidden
        self.hidden_dim = self.flows[0].theta.hidden_dim
        
          
    
    def configure_flows_field(n_flows,num_hidden,hidden_dim,lattice,p_drop=0):
        flows = []

        for k in range(n_flows//2):
            for dir in range(lattice.n_dims):
                split_masks = lattice.get_pair_split_masks_field(dir)
                theta = ThetaNetwork.configure_theta(num_hidden = num_hidden, 
                                                 hidden_dim = hidden_dim, 
                                                 p_drop = p_drop,
                                                 in_dim = lattice.total_nodes//2,
                                                 out_dim = lattice.total_nodes//2)
                flows.append(AffineCouplingLayer(theta, split = split_masks, swap = k % 2))
   
        flows = nn.ModuleList(flows)
        return flows     
    
    def config_and_init(n_flows,num_hidden,hidden_dim,lattice,ort):
        flows = NormalizingFlow.configure_flows_field(n_flows,num_hidden,hidden_dim,lattice)
        return NormalizingFlow(flows,lattice,ort)
    
    def save(self,filename):
        state_dict = self.state_dict()
        model_dict={"state_dict":state_dict,
                    "n_flows":self.n_flows,
                    "num_hidden":self.num_hidden,
                    "hidden_dim":self.hidden_dim,
                    "ort":self.ort,
                    "n_nodes":self.lattice.n_nodes,
                    "sizes":self.lattice.sizes}
        torch.save(model_dict,filename)
    
    def load_model(filename):   
        model_dict = torch.load(filename,map_location = device,weights_only=False)
        lattice = Lattice(model_dict["n_nodes"],model_dict["sizes"])
        model = NormalizingFlow.config_and_init(model_dict["n_flows"],
                                  model_dict["num_hidden"], 
                                  model_dict["hidden_dim"],
                                  lattice,
                                  model_dict["ort"])
        model.load_state_dict(model_dict["state_dict"])
        return model


    

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        x, sum_log_abs_det = z, torch.zeros(z.size(0)).to(z.device)
        for flow in reversed(self.flows):
            x, log_abs_det = flow.g(x,params)
            sum_log_abs_det += log_abs_det
        
        if self.ort:
            x = torch.matmul(x,self.lattice.ort_mat_t.to(x.device))
            
        return x, sum_log_abs_det
    
        
    def __len__(self) -> int:
        return len(self.flows)
    
    def f(self, x: torch.Tensor,params=torch.tensor([])) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            res = x.clone()
            if self.ort:
                res = torch.matmul(res,self.lattice.ort_mat.to(res.device))
        
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
        
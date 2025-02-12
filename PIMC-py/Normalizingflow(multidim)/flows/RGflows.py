import torch
from torch import nn
from typing import Callable, List, Tuple


from flows.theta import ThetaNetwork
from flows.Layers import AffineCouplingLayer
from transforms import get_split_masks
from transforms import get_pair_split_masks
from flows.NormalizingFlow import NormalizingFlow
from transforms import config_RG_masks, t_get_O
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RGflows(nn.Module):
    
    def __init__(self,nflist, 
                 masks, 
                 n_flows_dict,
                 num_hidden_dict,
                 hidden_dim_dict,
                 grids_no_grad = 0,
                 p_drop = 0,
                 sys_dim = 1,
                 O_latt = torch.tensor([])):
        
        super().__init__()
        self.nflist = nflist
        self.masks = masks
        self.n_flows_dict = n_flows_dict
        self.num_hidden_dict = num_hidden_dict
        self.hidden_dim_dict = hidden_dim_dict
        self.out_dim = sum(list(map(len,masks)))
        self.grids_no_grad = grids_no_grad
        self.O = (t_get_O(self.out_dim)).to(device)
        self.Ot = (torch.t(self.O)).to(device)
        self.p_drop = p_drop
        self.sys_dim = sys_dim
        self.O_latt = O_latt
        self.O_latt_t = torch.t(O_latt)
        
    def set_out_dim(self,out_dim):
        self.out_dim = out_dim
        O = (torch.tensor(get_O(self.out_dim)).float()).to(device)
        self.Ot = (torch.t(O)).to(device)
        
    def configure_RG_flows(masks,n_flows_dict,num_hidden_dict,hidden_dim_dict,p_drop=0,sys_dim = 1):  
        nflist=[]
        dim = len(masks[0])
        param_dim = 0
        for k in range(len(masks)):
            flows =  NormalizingFlow.configure_flows(n_flows = n_flows_dict[dim],
                                                     num_hidden = num_hidden_dict[dim],
                                                     hidden_dim = hidden_dim_dict[dim], 
                                                     dim = dim , 
                                                     param_dim = param_dim, 
                                                     p_drop = p_drop,
                                                     sys_dim = sys_dim)
            nflist.append(NormalizingFlow(flows = flows))
            param_dim += dim 
            if k!=0: dim *= 2 
        nflist = nn.ModuleList(nflist)        
        return nflist    
        
    def configure_RG_model(masks,n_flows_dict, num_hidden_dict, hidden_dim_dict, grids_no_grad = 0, p_drop = 0, sys_dim = 1, O_latt = torch.tensor([])):
        flows = RGflows.configure_RG_flows(masks, n_flows_dict, num_hidden_dict, hidden_dim_dict, p_drop=p_drop, sys_dim = sys_dim)
        return RGflows(flows, 
                       masks, 
                       n_flows_dict, 
                       num_hidden_dict, 
                       hidden_dim_dict, 
                       grids_no_grad = grids_no_grad, 
                       p_drop = p_drop, 
                       sys_dim = sys_dim, 
                       O_latt = O_latt)
    
    def save(self,filename):
        state_dict = self.state_dict()
        model_dict={"state_dict":state_dict,
                    "n_flows_dict":self.n_flows_dict,
                    "num_hidden_dict":self.num_hidden_dict,
                    "hidden_dim_dict":self.hidden_dim_dict,
                    "masks":self.masks,
                    "p_drop":self.p_drop,
                    "sys_dim":self.sys_dim,
                    "O_latt":self.O_latt}
        torch.save(model_dict,filename)
    
    def load_model(filename):   
        model_dict = torch.load(filename,map_location = device)
        model=RGflows.configure_RG_model(model_dict["masks"],
                                  model_dict["n_flows_dict"], 
                                  model_dict["num_hidden_dict"],
                                  model_dict["hidden_dim_dict"],
                                  p_drop = model_dict["p_drop"],
                                  sys_dim = model_dict["sys_dim"],
                                  O_latt = model_dict["O_latt"])
        model.load_state_dict(model_dict["state_dict"])
        return model

    def g(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        sum_log_abs_det = torch.zeros(z.size(0)).to(z.device)
        params=torch.tensor([]).to(z.device)
        
        for i in range(len(self.nflist)):
            x = z[:,self.masks[i],:]
            x, log_abs_det = ((self.nflist)[i]).g(x,params)
            sum_log_abs_det += log_abs_det
            params = torch.cat((params,x.detach()),dim=1)
            z[:,self.masks[i],:] = x          
        
        for _ in range(self.sys_dim):
            z[:,:,_] = torch.matmul(z[:,:,_],self.Ot.to(z.device))
        
        if self.O_latt.shape[0]>0:
            for _ in range(self.out_dim):
                z[:,_,:] = torch.matmul(z[:,_,:],self.O_latt_t.to(z.device))
        
        return z, sum_log_abs_det
    

    def g_gr_ng(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        sum_log_abs_det = torch.zeros(z.size(0)).to(z.device)
        params=torch.tensor([]).to(z.device)
        full_mask = []
        
        with torch.no_grad():
            for i in range(0,self.grids_no_grad):
                x = z[:,self.masks[i],:]
                x, log_abs_det = ((self.nflist)[i]).g(x,params)
                sum_log_abs_det += log_abs_det
                params = torch.cat((params,x.detach()),dim=-1)
                z[:,self.masks[i],:] = x
                full_mask += self.masks[i]
        
        for i in range(self.grids_no_grad,len(self.nflist)):
            x = z[:,self.masks[i],:]
            x, log_abs_det = ((self.nflist)[i]).g(x,params)
            sum_log_abs_det += log_abs_det
            params = torch.cat((params,x.detach()),dim=-1)
            z[:,self.masks[i],:] = x
            full_mask += self.masks[i]          
            if len(full_mask) == self.out_dim:
                full_mask.sort()
                z = torch.matmul(z[:,full_mask,:],self.Ot.to(z.device))    
                return z, sum_log_abs_det
        
        z = torch.matmul(z,self.Ot.to(z.device))    
        return z, sum_log_abs_det
    
    
    
    def f(self, x: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        with torch.no_grad():
            res=x.clone()
            sum_log_abs_det = torch.zeros(res.size(0)).to(res.device)
            params=torch.tensor([]).to(res.device)
            
            if self.O_latt.shape[0]>0:
                for _ in range(self.out_dim):
                    res[:,_,:] = torch.matmul(res[:,_,:],self.O_latt.to(z.device))
            
            for _ in range(self.sys_dim):
                res[:,:,_] = torch.matmul(res[:,:,_],self.O.to(res.device))
        
            for i in range(len(self.nflist)):
                z = res[:,self.masks[i],:]
                tmp = z.clone()
                z, log_abs_det = ((self.nflist)[i]).f(z,params)
                sum_log_abs_det += log_abs_det
                params = torch.cat((params,tmp.detach()),dim=-1)
                res[:,self.masks[i],:] = z    
        
        return res, sum_log_abs_det
    
    
        
    def log_prob(self,x):
        tmp = x.clone()
        z, lad = self.f(tmp)
        log_prob = -torch.sum(z**2/2,dim=-1)-lad
        return log_prob
    
    
    def g_samp(self, z: torch.Tensor,params=torch.tensor([])) -> torch.Tensor:
        
        sum_log_abs_det = torch.zeros(z.size(0)).to(z.device)
        params = torch.tensor([]).to(z.device)
        res = z.clone()
        
        for i in range(len(self.nflist)):
            x=z[:,self.masks[i]]
            res1, log_abs_det = ((self.nflist)[i]).g(x,params)
            sum_log_abs_det += log_abs_det
            params = torch.cat((params,x),dim=-1)
            res[:,self.masks[i]] = res1
        res = torch.matmul(res,self.Ot.to(z.device))    
        return res, sum_log_abs_det
    
        
    def __len__(self) -> int:
        return len(self.flows)
    
    def forward(self,z,grads=False):
        t=z.clone()
        with torch.no_grad():
            x, log_abs_det=self.g(t)
        
        return x, log_abs_det
    
    def set_type(self,type):
        if type == "float":
            self.float()
            self.O = self.O.float()
            self.Ot = self.Ot.float()
        elif type == "double":
            self.double()
            self.O = self.O.double()
            self.Ot = self.Ot.double()

    def append_aff(self,hidden_dim,num_hidden,num_aff):
        self.n_flows_dict[8] += 1 
        self.nflist[0].append_aff(hidden_dim,num_hidden,num_aff)

    
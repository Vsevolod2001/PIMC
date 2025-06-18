import time
import torch
import numpy as np
class Metropolis:
    def __init__(self,
                 system,
                 N_samp,
                 d,
                 val = lambda x: torch.tensor([1]),
                 N_sweep = 100,
                 log_per = 1000,
                 filename = "./trajs_and_corr/1.txt",
                 open_mode = "w"):
        
        self.system = system
        self.n_nod = system.n_nod
        self.dim = self.system.dim
        self.n_samp = N_samp
        self.S = 0
        self.filename = filename
        self.d = d
        self.N_sweep = N_sweep
        self.log_per = log_per
        self.ar = 0
        self.shift_dist = torch.distributions.Uniform(torch.zeros((self.n_nod)), torch.ones(self.n_nod))
        self.un = torch.distributions.Uniform(0, 1)
        self.val = val
        self.open_mode = open_mode
        self.res=[]
        self.times=[]
        
    def sweep(self,x):
        shift = self.shift_dist.sample((self.n_samp,)).to(x.device)
        y = x + self.d*(2*shift-1)
    
        S_new = self.system.Full_S(y).to(x.device) 
    
        dS = S_new - self.S
        prob = torch.exp(-dS)
        ind = self.un.sample((self.n_samp,)).to(x.device)<prob
    
        mask = ind.nonzero()
        if len(mask)>1:
            mask=mask.squeeze()
    
        if  len(mask)>0:
            self.S[mask] = S_new[mask]
            x[mask,:] = y[mask,:]  
        self.ar = torch.mean(ind.type('torch.FloatTensor'))
    
    
    def log(self,x):
        v = self.val(x)
        print(torch.mean(v),self.ar)
        self.res.append(v.cpu())
        self.times.append(time.time() - self.start)
    
    
    def init_state(self,x):
        self.start = time.time()
        self.S = self.system.Full_S(x).to(x.device)
        self.ar = 0
        self.res = []
        self.times = []
        
    
    def run(self,x):
        self.init_state(x)
        if len(self.filename)>0:
            f = open(self.filename,self.open_mode)
        
        for i in range(self.N_sweep):
            if i % self.log_per == 0 and i>0:
                self.log(x)   
            self.sweep(x)
        
        if len(self.filename)>0:    
            X = x.clone()
            X = torch.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
            np.savetxt(f,X.numpy())    
            f.close()        
        return x    
    
import time
import torch
import numpy as np
class Langevin:
    def __init__(self,
                 system,
                 N_samp,
                 val = "none",
                 eps = 0.001,
                 N_sweep = 10,
                 log_per = 1000,
                 filename = "./trajs_and_corr/1.txt",
                 open_mode = "w"):
        self.system = system
        self.eps = eps
        self.N_sweep = N_sweep
        self.log_per = log_per
        self.n_nod = self.system.lattice.total_nodes
        self.n_samp = N_samp
        self.shift_dist = torch.distributions.Normal(torch.zeros(self.n_nod), torch.ones(self.n_nod))
        self.open_mode = open_mode
        self.filename = filename
        self.val = val
        self.res = []
        self.times = []
    

    def sweep(self,x):
        dw = self.shift_dist.sample((self.n_samp,)).to(x.device)
        x += 0.5*self.eps*self.system.F(x) + self.eps ** 0.5 * dw
        
    def log(self,x):
        if self.val!="none":
            print(self.val(x))
            self.res.append(torch.mean(self.val(x)).cpu())
            self.times.append(time.time()-self.start)
    
    
    def init_state(self):
        self.start=time.time()
        self.res=[]
        self.times=[]

    def run(self,x):
        self.init_state()

        if len(self.filename)>0:
            f = open(self.filename,self.open_mode)

        for i in range(self.N_sweep):
            if i%self.log_per==0:
                self.log(x)
            self.sweep(x)

        if len(self.filename)>0:    
            np.savetxt(f,x.cpu().numpy())    
            f.close()            
        return x        

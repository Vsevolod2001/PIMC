import time
import torch
import numpy as np
class NN_Metropolis:
    def __init__(self,
                 system,
                 N_samp,
                 model,
                 usual_sampler,
                 val = lambda x: torch.tensor([1.0]),
                 log_per = 1000,
                 filename="./trajs_and_corr/0.txt",
                 stat_filename="./trajs_and_corr/stat.txt"):
        
        self.system = system
        self.n_nod = system.lattice.total_nodes
        self.n_samp = N_samp
        self.S = 0
        self.model = model
        self.filename = filename
        self.log_per = log_per
        self.log_prob = torch.tensor([0.0])
        self.n_current_trajs = 0
        self.un = torch.distributions.Uniform(0, 1)
        self.latent = self.system.lattice.normal_sampler()
        self.val = val
        self.usual_sampler = usual_sampler
        self.stat_file = stat_filename
        self.count = 0
        self.device = self.system.lattice.device
        self.res=[]
        self.times=[]
        


    def sweep(self,x):
        with torch.no_grad():
            z_old , lad_f = self.model.f(x)
            llp_old = torch.sum(self.latent.log_prob(z_old.cpu())).to(self.device)
        
            z_new = self.latent.sample((1,)).to(self.device)
            llp_new = torch.sum(self.latent.log_prob(z_new.cpu())).to(self.device)
            x_new , lad_g = self.model.g(z_new)
        
        S_new = self.system.S(x_new).to(x.device) 
    
        dS = S_new-self.S

        self.log_prob = -dS+llp_old-llp_new+lad_g+lad_f
        prob = torch.exp(self.log_prob)
        self.is_accepted = self.un.sample((1,)).to(x.device)<prob
        if  self.is_accepted:
            self.S = S_new
            x = x_new     
    
    
    def log(self,x):
        v = torch.mean(self.val(x))
        self.res.append(v.cpu())
        self.times.append(time.time()-self.start)
    
    
    def init_state(self,x):
        self.start = time.time()
        self.S = self.system.S(x).to(x.device)
        self.ar = 0
        self.n_current_trajs = 0
        self.count = 0
        self.res = []
        self.times = []
        
    
    def run(self,x):
        self.init_state(x)
        f = open(self.filename,"w") 
        stat_f = open(self.stat_file,"w")   
        while self.n_current_trajs < self.n_samp:
            self.count += 1
            if self.n_current_trajs % self.log_per == 0:
                self.log(x)
            self.usual_sampler.run(x)    
            self.sweep(x)
            print("trajectories sampled: ",self.n_current_trajs,"log_prob: ",self.log_prob.cpu().item(),file=stat_f)

            if self.is_accepted:
                self.n_current_trajs += 1
                np.savetxt(f,x.detach().cpu().numpy())

        print("mean acceptance rate:", self.n_samp/self.count,file=stat_f)
        f.close()
        stat_f.close()        
        return x    
    
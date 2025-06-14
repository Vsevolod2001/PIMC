import time
import torch
import numpy as np
class NN_Metropolis:
    def __init__(self,
                 system,
                 N_samp,
                 model,
                 latent,
                 usual_sampler,
                 val = lambda x: 1,
                 log_per = 1000,
                 filename="./trajs_and_corr/0.txt"):
        
        self.system = system
        self.n_nod = system.n_nod
        self.dim = self.system.dim
        self.n_samp = N_samp
        self.S = 0
        self.model = model
        self.filename = filename
        self.log_per = log_per
        self.ar = 0
        self.n_current_trajs = 0
        self.un = torch.distributions.Uniform(0, 1)
        self.latent = latent
        self.val = val
        self.usual_sampler = usual_sampler
        self.res=[]
        self.times=[]
        


    def sweep(self,x):
        with torch.no_grad():
            z_old , lad_f = self.model.f(x)
            llp_old = torch.sum(self.latent.log_prob(z_old))
        
            z_new = self.latent.sample((1,))
            llp_new = torch.sum(self.latent.log_prob(z_new))
            x_new , lad_g = self.model.g(z_new)
        
        S_new = self.system.Full_S(x_new).to(x.device) 
    
        dS = S_new-self.S
        prob = torch.exp(-dS+llp_old-llp_new+lad_g+lad_f)
        self.is_accepted = self.un.sample((1,)).to(x.device)<prob
        if  self.is_accepted:
            self.S = S_new
            x = x_new   
        self.ar = prob  
    
    
    def log(self,x):
        v = torch.mean(self.val(x))
        print(self.n_current_trajs,self.ar,v)
        self.res.append(v.cpu())
        self.times.append(time.time()-self.start)
    
    
    def init_state(self,x):
        self.start = time.time()
        self.S = self.system.Full_S(x).to(x.device)
        self.ar = 0
        self.n_current_trajs = 0
        self.res = []
        self.times = []
        
    
    def run(self,x):
        self.init_state(x)
        f = open(self.filename,"w")    
        while self.n_current_trajs < self.n_samp:
            if self.n_current_trajs % self.log_per == 0:
                self.log(x)
            self.usual_sampler.run(x)    
            self.sweep(x)
            if self.is_accepted:
                self.n_current_trajs += 1
                X = torch.clone(x)
                X = torch.reshape(X,(1,X.shape[1]*X.shape[2]))
                np.savetxt(f,X.numpy())    
        f.close()        
        return x    
    
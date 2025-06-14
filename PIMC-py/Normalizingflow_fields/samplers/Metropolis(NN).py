import time
import torch
import numpy as np
class NN_Metropolis:
    def __init__(self,
                 system,
                 N_samp,
                 d,
                 model="none",
                 latent="none",
                 n_us = 10,
                 N_sweep = 100,
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
        self.shift_dist = torch.distributions.Uniform(torch.zeros((self.n_nod,self.dim)), torch.ones(self.n_nod,self.dim))
        self.un = torch.distributions.Uniform(0, 1)
        self.latent = latent
        self.res=[]
        self.times=[]
        


    def sweep(self,x):
        with torch.no_grad():
            z_old , lad_f = self.model.f(x)
            llp_old = torch.sum(self.latent.log_prob(z_old))
        
            z_new = self.latent.sample((self.n_samp,))
            llp_new = torch.sum(self.latent.log_prob(z_new))
            x_new , lad_g = self.model.g(z_new)
        
        S_new = self.system.Full_S(x_new).to(x.device) 
    
        dS=S_new-self.S
        prob = torch.exp(-dS+llp_old-llp_new+lad_g+lad_f)
        ind = self.un.sample((self.n_samp,)).to(x.device)<prob
        if  ind:
            self.S = S_new
            x = x   
        self.ar = prob  
    
    
    def log(self,x):
        print(torch.mean(x**2),self.ar,self.mean_ar)
        self.res.append(torch.mean(x**2).cpu())
        self.times.append(time.time()-self.start)
    
    
    def init_state(self,x):
        self.start = time.time()
        self.S = self.system.Full_S(x).to(x.device)
        self.ar = 0
        self.mean_ar = 0
        self.res = []
        self.times = []
        
    
    def run(self,x):
        
        self.init_state(x)
        f = open(self.filename,"w")
        else:    
            for i in range(self.N_sweep):
                
                if i % self.log_per == 0:
                    self.log(x)
                
                rr = 1
                
                while rr > 0:
                    self.sweepNN(x)
                    rr *= (1-self.ar)
                
                self.mean_ar=0
                for k in range(self.n_us):
                    self.sweep(x)
                    self.mean_ar += self.ar 
                self.mean_ar *= (1/self.n_us)
                np.savetxt(f,x.numpy())
        f.close()        
        return x    
    
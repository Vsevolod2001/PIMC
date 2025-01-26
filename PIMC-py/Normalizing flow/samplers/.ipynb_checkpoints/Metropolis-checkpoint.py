import time
import torch
import numpy as np
class Metropolis:
    def __init__(self,
                 system,
                 N_samp,
                 d,
                 model="none",
                 latent="none",
                 n_us=10,
                 N_sweep=100,
                 log_per=1000,
                 filename="./trajs_and_corr/0.txt"):
        
        self.system = system
        self.n_nod = system.n_nod
        self.n_samp = N_samp
        self.S = 0
        self.model = model
        self.filename = filename
        self.d = d
        self.N_sweep = N_sweep
        self.n_us = n_us
        self.log_per = log_per
        self.ar = 0
        self.mean_ar = 0
        self.shift_dist = torch.distributions.Uniform(torch.zeros(self.n_nod), torch.ones(self.n_nod))
        self.un = torch.distributions.Uniform(0, 1)
        self.latent = latent
        self.res=[]
        self.times=[]
        
    def sweep(self,x):
        shift = self.shift_dist.sample((self.n_samp,)).to(x.device)
        y = x+self.d*(2*shift-1)
    
        S_new = self.system.Full_S(y).to(x.device) 
    
        dS=S_new-self.S
        prob = torch.exp(-dS)
        ind = self.un.sample((self.n_samp,)).to(x.device)<prob
    
        mask=ind.nonzero()
        if len(mask)>1:
            mask=mask.squeeze()
    
        if  len(mask)>0:
            self.S[mask]=S_new[mask]
            x[mask,:] = y[mask,:]  
        self.ar = torch.mean(ind.type('torch.FloatTensor'))


    def sweepNN(self,x):
        with torch.no_grad():
            z_old , lad_f = self.model.f(x)
            llp_old = torch.sum(self.latent.log_prob(z_old),-1)
        
            z_new = self.latent.sample((self.n_samp,))
            llp_new = torch.sum(self.latent.log_prob(z_new),-1)
            x_new , lad_g = self.model.g(z_new)
        
        S_new = self.system.Full_S(x_new).to(x.device) 
    
        dS=S_new-self.S
        prob = torch.exp(-dS+llp_old-llp_new+lad_g+lad_f)
        #print(prob)
        ind = self.un.sample((self.n_samp,)).to(x.device)<prob
        # ind int tensor
        # self.rr *= ind
        mask=ind.nonzero()
        if len(mask)>1:
            mask=mask.squeeze()
    
        if  len(mask)>0:
            self.S[mask] = S_new[mask]
            x[mask,:] = x_new[mask,:]   
        self.ar = torch.mean(ind.type('torch.FloatTensor'))  
    
    
    def log(self,x):
        print(torch.mean(x**2),self.ar,self.mean_ar)
        self.res.append(torch.mean(x**2).cpu())
        self.Times.append(time.time()-self.start)
    
    
    def init_state(self,x):
        self.start = time.time()
        self.S = self.system.Full_S(x).to(x.device)
        self.ar = 0
        self.mean_ar = 0
        self.res = []
        self.Times = []
        
    
    def run(self,x):
        
        self.init_state(x)
        f = open(self.filename,"w")

        
        if self.model=="none":
            
            for i in range(self.N_sweep):
                if i%self.log_per==0:
                    self.log(x)   
                self.sweep(x)
            np.savetxt(f,x.numpy())    
        
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
    
import torch
from LOSS import KL_with_S
class System:
    def __init__(self,n_nod,beta,dim=1,**args):
        self.normalizer=0
        self.Log_Z=0
        self.n_nod=n_nod
        self.beta=beta
        self.a=beta/n_nod
        self.dim = dim
    
    def T(self,diff):
        return 0
    
    def V(self,x):
        return 0
  
    
    def Full_S(self,x):
        Full_S = 0
        x_next = torch.roll(x,-1,1)
        diff = x_next-x
        Full_T = torch.sum(self.T(diff),dim=1)
        Full_V = torch.sum(self.V(x),dim=1)
        Full_S = self.a*(Full_T+Full_V) + self.n_nod*self.normalizer + self.Log_Z
        return Full_S    
            
        
    def get_S(self):
        def S(x):
            return self.Full_S(x)
        return S
    
    def get_KL(self):
        S = self.get_S()
        KL = KL_with_S(S,self.n_nod,self.dim)
        return KL
    
    
    def F_V(self,x):    #U=-dV/dx
        return 0
    
    def F_T(self,diff):      #Y=dT(xi)/dxi
        return 0
    
    def F(self,x):   #F=-dS/dx
        a = self.a
        x_next = torch.roll(x,-1,1)
        x_prev = torch.roll(x,1,1)
        diff_next = x_next-x
        diff_prev = x-x_prev
        
        F = a * self.F_V(x) + a * self.F_T(diff_next) - a * self.F_T(diff_prev)
        return F
        
        
    
    
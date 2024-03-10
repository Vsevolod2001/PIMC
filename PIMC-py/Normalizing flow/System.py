import torch
from NFconstants import a, N_nod
from LOSS import KL_with_S
class System:
    def __init__(self,**args):
        self.normalizer=0
    
    def T(self,diff):
        return 0
    
    def V(self,x):
        return 0
    
    def S(self,currp,nextp):
        diff=currp-nextp
        return a * (self.T(diff) + self.V(currp)) / self.hbar
  
    
    def Full_S(self,x):
        Full_S=0
        x_next=torch.roll(x,-1,1)
        diff=x_next-x
        Full_T=torch.sum(self.T(diff),axis=1)
        Full_V=torch.sum(self.V(x),axis=1)
        Full_S=a*(Full_T+Full_V)+N_nod*self.normalizer
        return Full_S    
            
        
    def get_S(self):
        def S(x):
            return self.Full_S(x)
        return S
    
    def get_KL(self):
        S=self.get_S()
        KL=KL_with_S(S)
        return KL
    
    def make_KL(cls,**args):
        obj=cls(**args)
        KL=obj.get_KL()
        return KL
    
    
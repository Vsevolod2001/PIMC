import torch
from NFconstants import a, N_nod
class system:
    def __init__(self,hbar,m):
        self.m = m
        self.hbar = hbar
    
    def T(self,diff):
        return self.m * (diff) ** 2 / (2 * a ** 2)
    
    def V(self,currp):
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
        Full_S=Full_T+Full_V
        return Full_S    
            
        
    def get_S(self):
        def S(x):
            return self.Full_S(x)
        return S
    
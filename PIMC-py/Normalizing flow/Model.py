import torch
from constants import a, N_nod
class model:
    def __init__(self,hbar,m):
        self.m = m
        self.hbar = hbar
    
    def T(self,currp,nextp):
        return self.m * (currp-nextp) ** 2 / (2 * a ** 2)
    
    def V(self,currp):
        return 0
    
    def S(self,currp,nextp):
        return a * (self.T(currp,nextp) + self.V(currp)) / self.hbar
    
    def dS(self,prevp,currp,nextp,new):
        return a * (self.T(new,nextp) - self.T(currp,nextp) + self.T(prevp,new) - self.T(prevp,currp) + self.V(new) - self.V(currp)) /       self.hbar
    
    def Full_S(self,x):
        Full_S=0
        for i in range(N_nod):
            Full_S+=self.S(x[i],x[(i+1)%N_nod])
        return Full_S    
            
        
    def get_S(self):
        def S(x):
            return self.Full_S(x)
        return S
    
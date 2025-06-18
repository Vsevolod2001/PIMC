import torch
from LOSS import KL_with_S
from lattice import Lattice
class System:
    
    
    def __init__(self,lattice):
        self.lattice = lattice
    
  
    
    def S(self,x):
        return 0    
            
        
    def get_S(self):
        def s(x):
            return self.S(x)
        return s
    
    def get_KL(self):
        s = self.get_S()
        KL = KL_with_S(s,self.lattice)
        return KL
    
    
    def F(self,x):   #F=-dS/dx
        return 0
        
        
    
    
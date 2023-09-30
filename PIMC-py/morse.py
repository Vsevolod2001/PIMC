import numpy as np
from math import gamma
from Model import model
from constants import MASS, HBAR, G, Q

class morse(model):
    
    def __init__(self,hbar=HBAR,m=MASS,g=G,q=Q):
        super().__init__(m, hbar)
        self.g=g
        self.q=q
        
    
    def V(self,currp):
        z=np.exp(-self.q*currp)
        return (0.5*(self.g)**2)*((z-1)**2 - 1)
    
    def theor_Psi2(self,x):
        L=self.hbar/(self.m ** 0.5 * self.g)
        alpha=self.q*L
        z=x/L
        N=2**(1/alpha) * (2-alpha)/( (alpha ** (3-2 * alpha)) * gamma(2/alpha))
        power=(-2 / alpha) * np.exp(-alpha * z) - (2-alpha) * z
        P=N/L*np.exp(power)
        return P
basic_morse=morse(1,1,1,1)    

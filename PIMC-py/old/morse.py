import numpy as np
from math import gamma
from Model import model
from constants import MASS, HBAR, g0, q0

class morse(model):
    
    def __init__(self,hbar=HBAR,m=MASS,g=g0,q=q0):
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
        N=2**(2 / alpha - 1) * (2-alpha)/( (alpha ** (2/alpha - 1)) * gamma(2/alpha))
        power=(-2 / alpha) * np.exp(-alpha * z) - (2-alpha) * z
        P=N/L*np.exp(power)
        return P
basic_morse=morse(1,1,1,1)    

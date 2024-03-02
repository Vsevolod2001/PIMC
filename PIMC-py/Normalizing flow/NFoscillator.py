from System import system
from NFconstants import a
import numpy as np

class oscillator(system):
    
    def __init__(self,hbar=1,m=1,w=1):
        super().__init__(m, hbar)
        self.w=w
    
    def T(self,diff):
        return self.m * (diff) ** 2 / (2 * a ** 2)
        
    def V(self,x):
        return self.m * (self.w) ** 2 * x ** 2 / 2
    
    def theor_Psi2(self,x):
        L=(self.hbar/(self.m * self.w))**0.5
        z=x/L
        return (np.pi) ** (-0.5) * np.exp(-z ** 2) /L

basic_oscillator=oscillator()
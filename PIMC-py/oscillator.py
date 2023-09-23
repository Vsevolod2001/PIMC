from Model import model
from constants import MASS, OMEGA, HBAR
import numpy as np

class oscillator(model):
    
    def __init__(self,hbar=HBAR,m=MASS,w=OMEGA):
        super().__init__(m, hbar)
        self.w=w
        
    
    def V(self,currp):
        return self.m * (self.w) ** 2 * currp ** 2 / 2
    
    def theor_Psi2(self,x):
        L=(self.hbar/(self.m * self.w))**0.5
        z=x/L
        return (np.pi) ** (-0.5) * np.exp(-z ** 2) /L

basic_oscillator=oscillator()

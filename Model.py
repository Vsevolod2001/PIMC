from constants import a, MASS, OMEGA, HBAR
class model:
    def __init__(self,m=MASS,w=OMEGA,hbar=HBAR):
        self.m=m
        self.w=w
        self.hbar=hbar
    
    def T(self,currp,nextp):
        return self.m * (currp-nextp)**2 / (2 * a ** 2)
    def V(self,currp):
        return self.m * (self.w) ** 2 * currp ** 2 / 2
    def S(self,currp,nextp):
        return a * (self.T(currp,nextp) + self.V(currp)) / self.hbar

basic_model=model()    
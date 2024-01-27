from constants import a
class model:
    def __init__(self,hbar,m):
        self.m=m
        self.hbar=hbar
    
    def T(self,currp,nextp):
        return self.m * (currp-nextp)**2 / (2 * a ** 2)
    def V(self,currp):
        return 0
    def S(self,currp,nextp):
        return a * (self.T(currp,nextp) + self.V(currp)) / self.hbar
    def dS(self,prevp,currp,nextp,new):
        return a * (self.T(new,nextp)-self.T(currp,nextp)+self.T(prevp,new)-self.T(prevp,currp)+self.V(new)-self.V(currp)) / self.hbar

    
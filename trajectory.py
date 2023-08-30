import numpy as np
from constants import N_nod,a,d,n_att,D
from Model import model
import Value
class trajectory:

    def __init__(self,x,model):
        self.x=x
        self.model=model
    
    def __str__(self):
        return str(self.points) 

    def markov(self):
        mod=self.model
        dS=0
        currp=0
        prevp=0
        nextp=0
        new=0
        for j in range(N_nod):
            i=np.random.randint(0,N_nod)
            currp=self.x[i]
            prevp=self.x[(i-1)%N_nod]
            nextp=self.x[(i+1)%N_nod]
            for k in range(n_att):
                new=currp+(2*d)*np.random.rand()-d
                dS = mod.S(prevp,new)-mod.S(prevp,currp)+mod.S(new,nextp)-mod.S(currp,nextp)
                
                if dS<=0:
                    currp=new
                else:
                    r=np.random.rand()
                    if r<np.exp(-dS):
                        currp=new
            self.x[i]=currp   
        return 0         
    
    
    def average(self,value):
        Mean=0
        Np=value.N_points
        if Np==1:
            for i in range(N_nod):
                Mean+=value.func(self.x[i],self.model)
        
        if Np==2:
            for i in range(N_nod):
                Mean+=value.func( self.x[i%N_nod] , self.x[(i+1)%N_nod], self.model )
        return Mean/N_nod
    
    def randgen(model,method):
        
        if method=="warm":
            points=(2*D)*np.random.rand(N_nod)-D
        
        if method=="cold":
            points=N_nod*[0]
        
        return trajectory(points,model)
    
    def P(self,p,n_bins,x_left,x_right):
        h=(x_right-x_left)/n_bins
        for i in range(N_nod):
            k=int((self.x[i]-x_left)/h)
            if k<n_bins:
                p[k]+=1
        return 0 
        
import numpy as np
from constants import N_nod,a,d,n_att,D
from Model import model
import Value
from andist import O, diag_mat
class trajectory:

    def __init__(self,x,model):
        self.x=x
        self.model=model
    
    def __str__(self):
        return str(self.points) 
    
    Ind=list(range(N_nod))
    
    
    
    def markov_node(prevp,currp,nextp,R,r,model):
        new=0
        dS=0
        for k in range(n_att):
            new=currp+(2*d)*R[k]-d
            dS = model.dS(prevp,currp,nextp,new)
            if dS<=0:
                currp=new
            else:
                if r[k]<np.exp(-dS):
                    currp=new            
        return currp
        
    
    
    def markov(self):
        mod=self.model
        currp=0
        prevp=0
        nextp=0
        new=0
        dS=0
        np.random.shuffle(trajectory.Ind)
        R=np.random.rand(N_nod,n_att)
        r=np.random.rand(N_nod,n_att)
        for j in range(N_nod):
            i=trajectory.Ind[j]
            currp=self.x[i]
            prevp=self.x[(i-1)%N_nod]
            nextp=self.x[(i+1)%N_nod]
            self.x[i]=trajectory.markov_node(prevp,currp,nextp,R[j],r[j],mod)            
        return 0
    
    
    Vmarkov=np.frompyfunc(markov,1,1)
        
    def average(self,value):
        mean=0
        Np=value.N_points
        
        if Np==1:
            mean=np.mean(value.func(self.x,self.model))
        
        if Np==2:
            mean=np.mean(value.func( self.x , np.roll(self.x,value.shift), self.model) )    
        
        return mean
    
    Vaverage=np.frompyfunc(average,2,1)
    
    def randgen(model,method,y=1):
        
        if method=="warm":
            x=(2*D)*np.random.rand(N_nod)-D
        
        if method=="cold":
            x=np.zeros((N_nod))
        
        if method=="an":
            y=np.random.normal(size=(N_nod))
            y1=np.dot(diag_mat,y)
            x=np.dot(O,y1)
        
        return trajectory(x,model)
    
    
    Vrandgen=np.frompyfunc(randgen,3,1)
    
    def convert_to_array(self):
        return np.copy(self.x)
    
    Vconv=np.frompyfunc(convert_to_array,1,1)

    
    def P(self,p,n_bins,x_left,x_right):
        h=(x_right-x_left)/n_bins
        for i in range(N_nod):
            k=int((self.x[i]-x_left)/h)
            if k<n_bins:
                p[k]+=1
        return 0
    
        
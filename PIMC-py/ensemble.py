import trajectory
from constants import N_nod, N_traj, Bins, X_Left, X_Right
import numpy as np

class ensemble:
    def __init__(self,trajs,n_traj=N_traj):
        self.n_traj=n_traj
        self.trajs=trajs
        
    def randgen(model,method="cold",n_traj=N_traj):
        trajs=[0]*n_traj
        for i in range(n_traj):
            trajs[i]=trajectory.trajectory.randgen(model,method)
        return ensemble(trajs,n_traj)
    
    
    def average_and_sigma(self,value):
        Mean=0
        disp=0
        x=0
        for i in range(self.n_traj):
            x=(self.trajs[i]).average(value)
            Mean+=x
            disp+=x**2
        Mean=Mean/self.n_traj
        disp=(disp/self.n_traj) - Mean**2
        return np.array([Mean,disp**0.5])
    
    def markov(self):
        for i in range(self.n_traj):
            (self.trajs[i]).markov()
        return 0    
    
    def P(self,p,n_bins=Bins,x_left=X_Left,x_right=X_Right):
        for i in range(self.n_traj):
            (self.trajs[i]).P(p,n_bins,x_left,x_right)    
        return 0 
    
  

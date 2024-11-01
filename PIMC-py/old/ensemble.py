import trajectory
from constants import N_nod, N_traj, Bins, X_Left, X_Right
import numpy as np

class ensemble:
   
    def __init__(self,trajs,n_traj=N_traj):
        self.n_traj=n_traj
        self.trajs=trajs
    
    def randgen(model,method,n_traj=N_traj):
        trajs=np.zeros((n_traj),dtype=object)
        trajs=trajectory.trajectory.Vrandgen(model,method,trajs)
        return ensemble(trajs,n_traj)
        
    
    def average_and_sigma(self,value):
        x=trajectory.trajectory.Vaverage(self.trajs,value)
        mean=np.mean(x)
        std=np.std(x)
        return np.array([mean,std*(self.n_traj)**-0.5])
    
    Vaverage_and_sigma=np.frompyfunc(average_and_sigma,2,1)
   
    
    def markov(self):
        trajectory.trajectory.Vmarkov(self.trajs)
        return 0    
    
    def convert_to_array(self):
        return np.vstack(trajectory.trajectory.Vconv(self.trajs))
    
    def save(self,filename):
        np.savetxt(filename,self.convert_to_array(),delimiter=',')
        return 0
    
    def load(filename,model):
        X=np.loadtxt(filename,delimiter=',')
        n_traj=X.shape[0]
        trajs=np.zeros((n_traj),dtype=object)
        for i in range(n_traj):
            trajs[i]=trajectory.trajectory(X[i],model)
        return ensemble(trajs,n_traj)    
    
    def P(self,p,n_bins=Bins,x_left=X_Left,x_right=X_Right):
        for i in range(self.n_traj):
            (self.trajs[i]).P(p,n_bins,x_left,x_right)    
        return 0 
    
  
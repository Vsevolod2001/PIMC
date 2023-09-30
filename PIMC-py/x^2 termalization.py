import numpy as np
import matplotlib.pyplot as plt
from constants import N_nod,a,d,D,n_att,N_traj,meash,step,sweeps
from trajectory import trajectory
from Value import X_POW_2



def average_and_sigma(T,N_traj,value):
    Mean=0
    disp=0
    x=0
    for i in range(N_traj):
        x=T[i].average(value)
        Mean+=x
        disp+=x**2
    Mean=Mean/N_traj
    disp=disp/N_traj-Mean**2
    return np.array([Mean,disp**0.5])






T=[trajectory.randgen("cold") for i in range(N_traj)]

V=0
Varr=np.zeros((meash,2))
for j in range(1,sweeps+1):
    V=0
    print(j)
    for i in range(N_traj):
        T[i].markov()
    if j%step==0 or j==sweeps:    
        V=average_and_sigma(T,N_traj,X_POW_2)
        print(V)
        Varr[j//step-1]=V

    

Varr=np.transpose(Varr)

it=[(i+1)*step for i in range(meash)]
plt.figure()
plt.scatter(it,Varr[0],s=10)
plt.grid(True)
plt.axis([0,sweeps,0,1])
plt.show()

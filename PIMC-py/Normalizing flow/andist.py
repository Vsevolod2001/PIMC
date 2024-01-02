import numpy as np
from NFconstants import N_nod,a

N=N_nod
O=np.zeros((N,N))
for k in range(N):
    O[k][0]=1
Smax=(N-1-int(N%2==0))//2    
for s in range(1,Smax+1):
    for k in range(N):
        O[k][2*s-1]=(2**0.5)*np.cos(2*np.pi*s*k/N)
        O[k][2*s]=(2**0.5)*np.sin(2*np.pi*s*k/N)
if N%2==0:
    for k in range(N):
        O[k][N-1]=(-1)**k
O=O / (N ** 0.5)

n=[0]*N
n[0]=(1/a)**0.5
for s in range(1,Smax+1):
    ev=4 * np.sin(np.pi*s/N) ** 2+a**2
    f=(a/ev)**0.5
    n[2*s-1]=f
    n[2*s]=f

if N%2==0:
    n[N-1]=(a/(4+a**2))**0.5
diag_mat=np.diag(n)
C=np.dot(O,diag_mat)

A=np.zeros((N_nod,N_nod))
for i in range(N_nod):
    A[i][(i-1)%N_nod]=-1
    A[i][i]=2+a**2
    A[i][(i+1)%N_nod]=-1
A=A/a    
    
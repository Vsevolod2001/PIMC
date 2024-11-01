import numpy as np
import matplotlib.pyplot as plt
from constants import N_nod, N_traj, a, n_att, sweeps, D, d
def MNK(X,Y,S):
    if (len(X)!=len(Y)):
        print("Y",len(X),len(Y))
    if (len(X)!=len(S)):
        print("S",len(X),len(S))    
    x=[X[i]/S[i] for i in range(len(X))]
    y=[Y[i]/S[i] for i in range(len(X))]
    s=[1/S[i] for i in range(len(X))]
    xx=np.dot(x,x)
    ss=np.dot(s,s)
    xs=np.dot(x,s)
    xy=np.dot(x,y)
    ys=np.dot(s,y)
    d=xx*ss-(xs)**2
    da=xy*ss-xs*ys
    db=xx*ys-xs*xy
    a=da/d
    b=db/d
    Sa=(ss/d)**0.5
    Sb=(xx/d)**0.5
   
    MX=sum(X)/len(X)
    MY=sum(Y)/len(Y)
    X0=[X[i]-MX for i in range(len(X))]
    Y0=[Y[i]-MY for i in range(len(Y))]
    NX=(np.dot(X0,X0))**0.5
    NY=(np.dot(Y0,Y0))**0.5
    XY=np.dot(X0,Y0)
    R=XY/(NX*NY)
    
    delta=[Y[i]-a*X[i]-b for i in range(len(x))]
    E=np.dot(delta,delta)
    print("a=",a)
    print("Sa=",Sa)
    print("b=",b)
    print("Sb=",Sb)
    print("Rcorr=",R)
    E=(E/(len(X)))**0.5
    print("E=",E)
    print("\n")
    A=[a,b,Sa,Sb,E]
    return A

def f(x,a,b):
    return a*x+b

def graph(X,Y,S=0,Sx=0):
    A=MNK(X,Y,S)
    Z=[f(x,A[0],A[1]) for x in X]
    plt.scatter(X,Y)
    plt.errorbar(X,Y,xerr=Sx,yerr=S,fmt='none',ecolor="red")
    plt.plot(X,Z,color="black")
    return A

def graph_Probdens(dens,theor_dens,n_bins,x_left,x_right):
    h=(x_right-x_left)/n_bins
    x=np.array([x_left+i*h for i in range(n_bins)])
    theor=theor_dens(x)
    plt.figure()
    plt.ylabel(r"$|\psi|^2$",fontsize=17)
    plt.xlabel(r"$x$",fontsize=17) 
    plt.title("N_nod="+str(N_nod)+" N_traj="+str(N_traj)+" a="+str(a)+" n_att="+str(n_att)+" sweeps="+str(sweeps))
    plt.scatter(x,dens,s=10)
    plt.plot(x,theor)
    plt.grid(True)
    Max=max(np.max(dens),np.max(theor))
    plt.axis([x_left,x_right,0,1.5*Max])
    plt.show()

def graph_Termalization(Varr,meash,step):
    it=[(i+1)*step for i in range(meash)]
    plt.figure()
    plt.errorbar(it,Varr[0],xerr=0,yerr=Varr[1],fmt='none',ecolor="red")
    plt.scatter(it,Varr[0],s=10)
    plt.grid(True)
    sweeps=meash*step
    MIN=min(Varr[0])
    MAX=max(Varr[0])
    delta=MAX-MIN
    plt.axis([0,sweeps,MIN-0.1*delta,MAX+0.1*delta])
    plt.title("N_nod="+str(N_nod)+" N_traj="+str(N_traj)+" a="+str(a)+" n_att="+str(n_att)+" D="+str(D)+" d="+str(d))
    plt.show()
    
def graph_lindependence(parametrs,values,param_name,value_name,n_experiments):
    plt.figure()
    plt.xlabel(param_name,fontsize=17)
    plt.ylabel(value_name,fontsize=17)
    plt.title("N_nod="+str(N_nod)+" N_traj="+str(N_traj)+" a="+str(a)+" n_att="+str(n_att)+" sweeps="+str(sweeps))
    coeffs=graph(parametrs,values[0],values[1])
    plt.legend([value_name+"="+"("+str(coeffs[0])[:5]+r"$\pm$"+str(coeffs[2])[:5]+")"+param_name+"+"+"("+str(coeffs[1])[:5]+r"$\pm$"+str(coeffs[3])[:5]+")"])
    MAX=max(values[0])
    MIN=min(values[0])
    delta=MAX-MIN
    plt.axis([0,1.1*max(parametrs),MIN-0.05*delta,MAX+0.05*delta])
    plt.grid(True)
    plt.show()
    return coeffs
    

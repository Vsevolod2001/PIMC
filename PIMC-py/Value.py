from NFconstants import a, NG_points
import numpy as np

class Value:
    def __init__(self,func,N_points,shift=1):
        self.func=np.frompyfunc(func,N_points+1,1)
        self.N_points=N_points
        self.shift=shift

def x_pow_2(x,model):
    return x**2

def delta_x_pow_2(x,y,model):
    return (x-y)**2

def p_pow_2(x,y,model):
    return ((1/a)**2)*(a-(y-x)**2)

def test_corr(x,y,model):
    return 2*x*y*np.exp(a)

def Pot(x,model):
    return model.V(x)

def Kin(x,y,model):
    return 0.5 * (1/a ** 2) * (a*model.hbar-model.m*(x-y)**2)

def corr(x,y,model):
    return x*y

POT=Value(Pot,1)
X_POW_2=Value(x_pow_2,1)
DELTA_X_POW_2=Value(delta_x_pow_2,2)
P_POW_2=Value(p_pow_2,2)
TEST_CORR=Value(test_corr,2)
KIN=Value(Kin,2)
G=np.array([Value(corr,2,i) for i in range(0,NG_points)])

from systems.Fields.scalar_field import Scalar_Field
from systems.Fields.phi4 import Phi4

K_nod = 3
N_nod = 2 ** K_nod
Beta = 16
Space_dim = 1
Mass2 = 1
L_nod = 9
R = 4


scalar = Scalar_Field(N_nod,Beta,Space_dim,Mass2,L_nod,R)
phi4 = Phi4(N_nod,Beta,Space_dim,Mass2,L_nod,R,1/24)
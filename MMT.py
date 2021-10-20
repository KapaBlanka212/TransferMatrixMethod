# Cod for transfer matrix method
# TODO: realisation matrix P and D : DONE
import MaterialConstant
import cmath
import sympy as sp
import numpy as np
from sympy import Symbol, simplify
from cmath import exp
from numpy import *
#==========================================================#
#                     CONSTANT                             #
#==========================================================#
m0 = 9.1*10**(-31) # electron mass kg
me = 0.35 * m0 # electron mass in thin membrane
const = MaterialConstant.MaterialConstant()
c = const.lightspeed
d_K108 = const.d_K108
l = const.full_wawelenght*10**(-2)
w = 2*pi*c*(const.full_wawelenght*(10**(-2))) # sec^(-1)
#print(l.shape)
n_K108 = const.n_K108
k_K108 = (const.alfa_K108)*(l/(4*pi))
n_vac = const.n_vac
k_vac = const.alfa_vac # 0 == 0
i = 1j # im 1
#==========================================================#
#                     THEORY DRUDE                         #
#==========================================================#
def eps(eps_inf,t):
    def w_p(Ne):
        w_p_return = math.sqrt((4*pi*Ne)/(eps_inf*me))
        return w_p_return
    eps_return = eps_inf * (1.0 - ((w_p**(2))/(w(w+(1j/t)))))
    return eps_return
def n_m(eps):
    n_m_return = number.real(eps)
    return n_m_return
def k_m(eps):
    k_m_return = number.imag(eps)
    return k_m_return
#==========================================================#
#     Complex refractive index and electric potential      #
#==========================================================#
# TODO: realisation MMT for only one point l/w : DONE
def n_(n,k):
    n_return = n+(1j)*k
    return n_return
def fi(w,n_,d):
    fi_out = (w/c)*n_*d
    return fi_out
#==========================================================#
#                  D and P matrix                          #
#==========================================================#
nm = Symbol('n')
km = Symbol('k')
dm = Symbol('d')
n_m = nm +i*km
# for all point (problem with symbols math)
# 0-1
#D0 = (1/2)*(1/n_(nm,km))*sp.Matrix([[(n_(nm,km)+n_vac),*(n_(nm,km)-n_vac)],
#                                    [(n_(nm,km)-n_vac),*(n_(nm,km)+n_vac)]])
# 1
#P_m = sp.Matrix([[sp.exp(1j*fi(n_(nm,km),dm)),0],
#                [0,sp.exp(-1j*fi(n_(nm,km),dm))]])
# 1-2
#D1 = (1/(2*n_(n_K108,k_K108)))*sp.Matrix[[*(n_(nm,km)+n_(n_K108,k_K108)),(n_(n_K108,k_K108)-n_(nm,km))],
#                                         [(n_(n_K108,k_K108)-n_(nm,km)),(n_(nm,km)+n_(n_K108,k_K108))]])
# 2
#P_K108 = np.matrix([[np.exp(1j*fi(n_(n_K108,k_K108),d_K108)),0],
#                   [0,np.exp(-1j*fi(n_(n_K108,k_K108),d_K108))]])
# 2-3
#D2 = (1/(2*n_vac))*sp.Matrix([[(n_vac+n_(n_K108,k_K108)),(n_vac - n_(n_K108,k_K108))],
#                             [(n_vac-n_(n_K108,k_K108)),(n_vac+n_(n_K108,k_K108))]])

# for one point

# 0-1
D0 = (1/2)*(1/n_m)*sp.Matrix([[n_m + n_vac[0],n_m - n_vac[0]],
                              [n_m - n_vac[0],n_m + n_vac[0]]])
# 1
P_m = sp.Matrix([[sp.exp(i*fi(w[0],n_(nm,km),dm)),0],
                [0,sp.exp(-i*fi(w[0],n_(nm,km),dm))]])
# 1-2
D1 = (1/(2*n_(n_K108[0],k_K108[0])))*sp.Matrix([[n_m+n_(n_K108[0],k_K108[0]),n_(n_K108[0],k_K108[0])-n_m],
                                               [n_(n_K108[0],k_K108[0])-n_m,n_m+n_(n_K108[0],k_K108[0])]])
# 2
P_K108 = sp.Matrix([[cmath.exp(i*fi(w[0],n_(n_K108[0],k_K108[0]),d_K108)),0],
                   [0,cmath.exp(-i*fi(w[0],n_(n_K108[0],k_K108[0]),d_K108))]])
# 2-3
D2 = (1/(2*n_vac[0]))*sp.Matrix([[n_vac[0]+n_(n_K108[0],k_K108[0]),n_vac[0] - n_(n_K108[0],k_K108[0])],
                                 [(n_vac[0]-n_(n_K108[0],k_K108[0])),n_vac[0]+n_(n_K108[0],k_K108[0])]])
#print(D0)
#print('')
#print(P_m)
#print('')
#print(D1)
#print('')
#print(P_K108)
#print('')
#print(D2)
#print('')

#==========================================================#
#                  CALCULATING T AND R                     #
#==========================================================#
M = D2@P_K108@D1@P_m@D0
T = abs(M[0,0]-(M[0,1]*M[1,0])/M[1,1])
T = simplify(T)
R = abs(M[1,0]/M[1,1])**(2)
R = simplify(R)
print('T ==',T)
print('R ==',R)
# Cod for transfer matrix method
# TODO: realisation matrix P and D
import math
import numbers
import numpy as np
import sympy as sp
import MaterialConstant
from sympy import Symbol
from math import *
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
#print(l,l.shape)
k_K108 = (const.alfa_K108)*(l/(4*pi))
w = 2*pi*c*(const.full_wawelenght*(10**(-2))) # sec^(-1)
#print(w)
A = (w/c)
#==========================================================#
#                     THEORY DRUDE                         #
#==========================================================#


def eps(eps_inf,t):
    def w_p(Ne):
        w_p_return = math.sqrt((4*pi*Ne)/(eps_inf*me))
        return w_p_return
    eps_return = eps_inf * (1.0 - ((w_p**(2))/(w(w+(1j/t)))))
    return eps_return


def n_m():
    n_m_return = number.real(eps)
    return n_m_return


def k_m():
    k_m_return = number.imag(eps)
    return k_m_return


#==========================================================#
#                   CREATE TRANSFER MATRIX                 #
#==========================================================#

#i = np.array([0,1,2,3])
def n_(n,k):
    n_return = n+(1j)*k
    return n_return
def fi(n_,d):
    fi_out = A*n_*d
    return fi_out


dm = Symbol('d')
dm = dm * np.ones((1,const.indx))
nm = Symbol('n_')
nm = nm * np.ones((1,const.indx))
expr = sp.exp(1j*fi(nm,dm))
expr2 = sp.exp(-1j*fi(nm,dm))

#print(nm+const.n_vac)
P_m = np.array([[expr,0],[0,expr2]])
P_K108 = np.array([[np.exp(1j*fi(n_(const.n_K108,k_K108),d_K108)),0],[0,np.exp(-1j*fi(n_(const.n_K108,k_K108),d_K108))]])
D0 = (1/(2*nm))*np.array([[nm+const.n_vac,nm-const.n_vac],[nm-const.n_vac,nm+const.n_vac]])
D1 = (1/(2*n_(const.n_K108,k_K108)))*np.array([[nm+n_(const.n_K108,k_K108),n_(const.n_K108,k_K108)],[n_(const.n_K108,k_K108)-nm,nm+n_(const.n_K108,k_K108)]])
D2 = (1/(2*const.n_vac))*np.array([[const.n_vac+n_(const.n_K108,k_K108),const.n_vac - n_(const.n_K108,k_K108)],[const.n_vac-n_(const.n_K108,k_K108),const.n_vac+n_(const.n_K108,k_K108)]])

#x = Symbol('x')
#print(fi(nm,dm))
#expr = exp(x)
#print(expr)

#P_m = np.matrix([[exp(1j*fi),0],[0,exp(-1j*fi)]])
#P = np.array(P_m,P_K108)
#def P():
#    P_return = np.matrix([[exp(1j*fi[i]),0],[0,exp(-1j*fi[i])]])
#    return P_return
#def D(n_):
#    D_return = (1/(2*n_[i+1]))*np.matrix([[n_[i+1]+n_[i],n_[i+1]-n_[i]],[n_[i+1]-n_[i],n_[i+1]+n_[i]]])
#    return D_return

#==========================================================#
#                  CALCULATING T AND R                     #
#==========================================================#

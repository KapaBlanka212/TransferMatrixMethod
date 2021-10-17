# Cod for transfer matrix method
# TODO: realisation matrix P and D
import math
import numbers

import MaterialConstant
from cmath import *
from math import pi
import numpy as np
from numpy import *
#==========================================================#
#                     CONSTANT                             #
#==========================================================#
m0 = 9.1*10**(-31) # electron mass kg
me = 0.35 * m0 # electron mass in thin membrane
const = MaterialConstant.MaterialConstant()
c = const.lightspeed
l = const.full_wawelenght*10**(-2)
#print(l,l.shape)
k_K108 = (const.alfa_K108)*(l/(4*pi))
w = 2*pi*c*(const.full_wawelenght*(10**(-2))) # sec^(-1)
#print(w)
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

i = np.array([0,1,2,3])

def n_(n,k):
    n_return = n[i]+(1j)*k[i]
    return n_return

def fi(n_,d):
    fi_return = (w/c)*n_[i]*d[i]
    return fi_return

def P():
    P_return = np.matrix([[exp(1j*fi[i]),0],[0,exp(-1j*fi[i])]])
    return P_return

def D(n_):
    D_return = (1/(2*n_[i+1]))*np.matrix([[n_[i+1]+n_[i],n_[i+1]-n_[i]],[n_[i+1]-n_[i],n_[i+1]+n_[i]]])
    return D_return


#==========================================================#
#                  CALCULATING T AND R                     #
#==========================================================#
#print(const.n_vac)
#n = np.array([const.n_vac,n_m,const.n_K108,const.n_vac],object)
#k = np.array([const.alfa_vac,k_m,k_K108,const.alfa_vac],object)
#d = np.array([])
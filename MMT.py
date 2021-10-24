# Cod for transfer matrix method
# TODO: realisation matrix P and D : DONE
import time
import MaterialConstant
import numpy as np
from numpy import exp,matrix,array,real,imag,sqrt,pi,transpose
#==========================================================#
#                     CONSTANT                             #
#==========================================================#
m0 = 9.1*10**(-31) # electron mass kg
me = 0.35 * m0 # electron mass in thin membrane
e0 = 1.6*10**(-19) # Kulon
const = MaterialConstant.MaterialConstant()
c = const.lightspeed*10**(-2) # cm/c
d_K108 = const.d_K108 #  cm
l = const.full_wavelenght # cm
w = 2*pi*c*(l) # sec^(-1)
n_K108 = const.n_K108 # K108
k_K108 = (const.alfa_K108)*(l/(4*pi)) # K108
n_vac = const.n_vac # VACUUM
k_vac = const.alfa_vac # 0 == 0
i = 1j # Im 1
indx = const.indx
#==========================================================#
#                 EXPERIMENTAL CONSTANT                    #
#==========================================================#
T_exp = const.T_exp
R_exp = const.R_exp
#==========================================================#
#                   CHANGES CONSTANT                       #
#==========================================================#
dm = 4*10**(-5) # m # 5*10**(-5)
Ne =  10 ** (20) # cm^-3 # 10 ** (21)
t = 10**(14) # 1/tau # 10*(15)
eps_Inf = 3 # 5
#==========================================================#
#                     THEORY DRUDE                         #
#==========================================================#
def eps(eps_inf,w_p,t,w): # t = 1/tau
    eps_return = eps_inf * (1.0 - ((w_p**(2))/(w*(w+(i*t)))))
    return eps_return
def w_p(Ne,eps_inf):
    w_p_out = np.sqrt((4*pi*Ne*e0**(2))/(eps_inf*me))
    return w_p_out
def n_m(eps):
    n_m_return = real(np.sqrt(eps))
    return n_m_return
def k_m(eps):
    k_m_return = imag(np.sqrt(eps))
    return k_m_return
#==========================================================#
#     Complex refractive index and electric potential      #
#==========================================================#
# TODO: realisation MMT for only one point l/w : DONE
def n_(n,k):
    n_return = n+(i)*k
    return n_return
def fi(w,n_,d):
    fi_out = (w/c)*n_*d
    return fi_out
#==========================================================#
#                  CALCULATING T AND R                     #
#==========================================================#
start1 = time.time()
wp = w_p(Ne, eps_Inf) # omega plasmone
Tm = [] # list of T
Rm = [] # list of R
def MMT(x):
    for m in range(0,x):
        eps_ = eps(eps_Inf, wp, t, w[m])
        nm = n_m(eps_)
        km = k_m(eps_)
        D0 = (1 / 2) * (1 / nm) * np.matrix([[nm + n_vac[m], nm - n_vac[m]],
                                             [nm - n_vac[m], nm + n_vac[m]]])
        # 1
        P_m = np.matrix([[np.exp(i * fi(w[m], n_(nm, km), dm)), 0],
                         [0, np.exp(-i * fi(w[m], n_(nm, km), dm))]])
        # 1-2
        D1 = (1 / (2 * n_(n_K108[m], k_K108[m]))) * np.matrix([[nm + n_(n_K108[m], k_K108[m]), n_(n_K108[m], k_K108[m]) - nm],
                                                               [n_(n_K108[m], k_K108[m]) - nm, nm + n_(n_K108[m], k_K108[m])]])
        # 2
        P_K108 = np.matrix([[np.exp(i * fi(w[m], n_(n_K108[m], k_K108[m]), d_K108)), 0],
                            [0, np.exp(-i * fi(w[m], n_(n_K108[m], k_K108[m]), d_K108))]])
        # 2-3
        D2 = (1 / (2 * n_vac[m])) * np.matrix([[n_vac[m] + n_(n_K108[m], k_K108[m]), n_vac[m] - n_(n_K108[m], k_K108[m])],
                                               [(n_vac[m] - n_(n_K108[m], k_K108[m])),n_vac[m] + n_(n_K108[m], k_K108[m])]])
        M = D2 @ P_K108 @ D1 @ P_m @ D0
        T = abs(M[0,0]-(M[0,1]*M[1,0])/M[1,1])
        Tm.append(T)
        R = abs(M[1,0]/M[1,1])**(2)
        Rm.append(R)
    anser = np.array([[Tm],[Rm]])

    return anser
#print(MMT(indx))
end1 = time.time()
print(end1 - start1)

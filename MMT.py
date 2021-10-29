import time
import MaterialConstant
import numpy as np
from numpy import *
import scipy as sp
from sympy import Symbol
from scipy.optimize import minimize
import scipy.integrate as integrate
# ==========================================================#
#                      CONSTANT                             #
# ==========================================================#
m0 = 9.1 * 10 ** (-31)  # electron mass kg
me = 0.35 * m0  # electron mass in thin membrane
e0 = 1.6 * 10 ** (-19)  # Kulon
const = MaterialConstant.MaterialConstant()
c = const.lightspeed * 10 ** (-2)  # cm/c
d_K108 = const.d_K108  # cm
l = const.full_wavelenght  # cm
w = 2 * pi * c / (l)  # sec^(-1)
n_K108 = const.n_K108  # K108
k_K108 = (const.alfa_K108) * (l / (4 * pi))  # K108
n_vac = const.n_vac  # VACUUM
k_vac = const.alfa_vac  # 0 == 0
i = 1j  # Im 1
indx = const.indx
indx2 = const.indx2
# ==========================================================#
#                 EXPERIMENTAL CONSTANT                    #
# ==========================================================#
T_exp = const.T_exp
R_exp = const.R_exp
full_exp = np.vstack([T_exp, R_exp])
# ==========================================================#
#           ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#
d_0 = 400 * 10 ** (-5)  # sm # 500*10**(-5)
Ne_0 = 10 ** (20)  # cm^-3 # 10 ** (21)
t_0 = 10 ** (14)  # 1/tau # 10*(15)
eps_Inf_0 = 3.0  # 5
# ==========================================================#
#                     THEORY DRUDE                         #
# ==========================================================#


def eps(eps_inf, w_p, t):  # t = 1/tau
    eps_return = eps_inf * (1.0 - ((w_p ** (2)) / (w * (w + (i * t)))))
    return eps_return


def w_p(Ne, eps_inf):
    w_p_out = np.sqrt((4 * pi * Ne * e0 ** (2)) / (eps_inf * me))
    return w_p_out


def n_m(eps):
    n_m_return = real(np.sqrt(eps))
    return n_m_return


def k_m(eps):
    k_m_return = imag(np.sqrt(eps))
    return k_m_return


# ==========================================================#
#     Complex refractive index and electric potential      #
# ==========================================================#
# TODO: realisation MMT for only one point l/w : DONE


def n_(n, k):
    n_return = n + i * k
    return n_return


def fi(w, n_, d):
    fi_out = (w / c) * n_ * d
    return fi_out


def fi_(w, n_, d, delta):
    fi_out = (w / c) * n_ * d + delta/2
    return fi_out


# ==========================================================#
#                  CALCULATING T AND R                     #
# ==========================================================#
start1 = time.time()
d_f = lambda : d_f
Ne_f = lambda : Ne_f
t_f = lambda : t_f
eps_Inf_f = lambda : eps_Inf_f
x = np.array([Ne_f,eps_Inf_f,t_f,d_f]).transpose()
x0 = np.array([Ne_0,eps_Inf_0,t_0,d_0]).transpose()


def mmt(x,delta):
    z = indx2
    Tm = []  # list of T
    Rm = []  # list of R
    for m in range(0, z):
        wp = w_p(x[0], x[1])  # omega plasmon
        eps_ = eps(x[1], wp, x[2])
        nm = n_m(eps_[m])
        km = k_m(eps_[m])
        D0 = (1 / 2) * (1 / nm) * np.array([[nm + n_vac[m], nm - n_vac[m]],
                                            [nm - n_vac[m], nm + n_vac[m]]])
        # 1
        P_m = np.array([[np.exp(i * fi(w[m], n_(nm, km), x[3])), 0],
                        [0, np.exp(-i * fi(w[m], n_(nm, km),x[3]))]])
        # 1-2
        D1 = (1 / (2 * n_(n_K108[m], k_K108[m]))) * np.matrix([[nm + n_(n_K108[m], k_K108[m]), n_(n_K108[m], k_K108[m]) - nm],
                                                                [n_(n_K108[m], k_K108[m]) - nm, nm + n_(n_K108[m], k_K108[m])]])
        # 2
        P_K108 = np.array([[(np.exp(i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))), 0],
                            [0, np.exp(-i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))]])
        # 2-3
        D2 = (1 / (2 * n_vac[m])) * np.array([[n_vac[m] + n_(n_K108[m], k_K108[m]), n_vac[m] - n_(n_K108[m], k_K108[m])],
                                            [(n_vac[m] - n_(n_K108[m], k_K108[m])), n_vac[m] + n_(n_K108[m], k_K108[m])]])
        M = D2 @ P_K108 @ D1 @ P_m @ D0
        T = (1/(2*pi))*integrate.quad(lambda : delta, abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1]) ** (2), 0, 2*pi)
        Tm.append(T)
        R = (1/(2*pi))*integrate.quad(lambda: delta, abs(M[1, 0] / M[1, 1]) ** (2),0, 2*pi)
        Rm.append(R)
    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack([[T1], [R1]])
    return eqn1


# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#
# TODO: find optimal parameters d,Ne,t,eps_Inf : in progress


def func(x,delta):
    S = mmt(x,delta) - full_exp
    S1 = np.std(S)
    return S1

delta = lambda : delta
bnds = ((Ne_0/10,None),(eps_Inf_0/300,None),(t_0/10,None),(d_0/10,None))
ans = sp.optimize.minimize(func,x0,method = 'Nelder-Mead',args = delta,
                           bounds=bnds,options={'disp':True,'return_all': True})
print(ans)
print(mmt((ans.x)),l*1000)
#x1 = np.array([1.17055461e+20, 1.30003878e+00, 1.23917067e+14, 4.30471629e-03])
#print([[mmt(x)],[l*1000]])
end1 = time.time()
print(end1 - start1)

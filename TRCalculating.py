import time
import MaterialConstant
import numpy as np
from numpy import *
import scipy as sp
#from sympy import Symbol
from scipy.optimize import minimize
import scipy.integrate as integrate
# ==========================================================#
#                     GLOBAL CONSTANT                       #
# ==========================================================#
m0 = 9.1 * 10 ** (-31)  # electron mass kg
me = 0.35 * m0  # electron mass in thin membrane
e0 = 1.6 * 10 ** (-19)  # Kulon
const = MaterialConstant.MaterialConstant()
c = const.lightspeed * 10 ** (-2)  # cm/c
d_K108 = const.d_K108  # cm
l = const.short_wavelenght[::10]  # cm
w = 2 * pi * c / (l)  # sec^(-1)
n_K108 = const.short_n[::10]
k_K108 = (const.short_alfa[::10]) * (l / (4 * pi))
n_vac = const.n_vac  # VACUUM
k_vac = const.alfa_vac  # 0 == 0
i = 1j  # Im 1
# ==========================================================#
#                  EXPERIMENTAL CONSTANT                    #
# ==========================================================#
T_exp = const.T6_short
R_exp = const.R6_short
indx = R_exp.size
full_exp = np.vstack([T_exp, R_exp])
# ==========================================================#
#           ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#
d_0 = 400 * 10 ** (-7)  # sm # 500*10**(-7)
Ne_0 = 10 ** (20)  # cm^-3 # 10 ** (21)
t_0 = 10 ** (14)  # 1/tau # 10*(15)
eps_Inf_0 = 3.0  # 5
# ==========================================================#
#                      THEORY DRUDE                         #
# ==========================================================#


def eps(eps_inf, wp, t):  # t = 1/tau
    eps_return = eps_inf * (1.0 - ((wp ** 2) / (w * (w + (i * t)))))
    return eps_return


def w_p(ne, eps_inf):
    w_p_out = np.sqrt((4 * pi * ne * e0 ** (2)) / (eps_inf * me))
    return w_p_out


def n_m(eps):
    n_m_return = real(np.sqrt(eps))
    return n_m_return


def k_m(eps):
    k_m_return = imag(np.sqrt(eps))
    return k_m_return


# ==========================================================#
#      Complex refractive index and electric potential      #
# ==========================================================#


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
#                   CALCULATING T AND R                     #
# ==========================================================#
start1 = time.time()


def x(ne_f, eps_inf_f, t_f, d_f):
    x_out = np.array([ne_f,eps_inf_f,t_f,d_f]).transpose()
    return x_out


x0 = np.array([0.5*10**21,4,0.5*10**15,450*10**(-7)]).transpose()


def mmt(x):
    z = indx
    Tm = []  # list of T
    Rm = []  # list of R
    for m in range(0, z):
        wp = w_p(x[0], x[1])  # omega plasmon
        eps_ = eps(x[1], wp, x[2])
        nm = n_m(eps_[m])
        km = k_m(eps_[m])
        D0 = (1 / 2) * (1 / nm) * np.array([[nm + n_vac, nm - n_vac],
                                            [nm - n_vac, nm + n_vac]])
        # 1
        P_m = np.array([[np.exp(i * fi(w[m], n_(nm, km), x[3])), 0],
                        [0, np.exp(-i * fi(w[m], n_(nm, km),x[3]))]])
        # 1-2
        D1 = (1 / (2 * n_(n_K108[m], k_K108[m]))) * np.array([[nm + n_(n_K108[m], k_K108[m]), n_(n_K108[m], k_K108[m]) - nm],
                                                                [n_(n_K108[m], k_K108[m]) - nm, nm + n_(n_K108[m], k_K108[m])]])
        # 2
        def P_K108(delta):
            P_K108_out = np.array([[np.exp(i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta)), 0],
                                [0,np.exp(-i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))]])
            return P_K108_out
        # 2-3
        D2 = (1 / (2 * n_vac)) * np.array([[n_vac + n_(n_K108[m], k_K108[m]), n_vac - n_(n_K108[m], k_K108[m])],
                                            [(n_vac - n_(n_K108[m], k_K108[m])), n_vac + n_(n_K108[m], k_K108[m])]])


        def matrix(delta):
            M_out = D2 @ P_K108(delta) @ D1 @ P_m @ D0
            return M_out


        def integrateT(delta):
            M = matrix(delta)
            T_d = abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1]) ** (2)
            return T_d

        T1 = integrate.quad(integrateT, 0, 2 * pi)
        T = (1 / (2 * pi)) * T1[0]
        Tm.append(T)

        def integrateR(delta):
            M = matrix(delta)
            R_d = abs(M[1, 0] / M[1, 1]) ** (2)
            return R_d


        R1 = integrate.quad(integrateR, 0, 2 * pi)
        R = (1 / (2 * pi)) * R1[0]
        Rm.append(R)

    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack([[T1], [R1]])
    return eqn1


# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#
# TODO: find optimal parameters d,Ne,t,eps_Inf : in progress


def func(x):
    s = mmt(x) - full_exp
    s1 = np.std(s[0,:])
    s2 = np.std(s[1,:])
    fun = s1 + s2
    return fun


bnds = ((Ne_0,10**21),
        (eps_Inf_0,5),
        (t_0,10**15),
        (d_0,500*10**(-7)))
ans = sp.optimize.minimize(func,x0,method = 'Nelder-Mead',bounds=bnds,
                           options={'disp':True,'return_all': None,'xatol': 0.01,
                                    'fatol': 0.01, 'maxiter':5000,'adaptive': True})
print(ans)
x_res = ans.x
TR = np.array(mmt(x_res)).transpose()
l1 = l * 10 ** 4
np.savetxt('TR',TR)
np.savetxt('L',l1)
end1 = time.time()
print(end1 - start1)

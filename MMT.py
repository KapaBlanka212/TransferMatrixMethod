# Cod for transfer matrix method
# TODO: realisation matrix P and D : DONE
import random
import time
import MaterialConstant
import numpy as np
from numpy import *
import scipy as sp
from scipy.optimize import curve_fit

# ==========================================================#
#                     CONSTANT                             #
# ==========================================================#
m0 = 9.1 * 10 ** (-31)  # electron mass kg
me = 0.35 * m0  # electron mass in thin membrane
e0 = 1.6 * 10 ** (-19)  # Kulon
const = MaterialConstant.MaterialConstant()
c = const.lightspeed * 10 ** (-2)  # cm/c
d_K108 = const.d_K108  # cm
l = const.full_wavelenght  # cm
w = 2 * pi * c * (l)  # sec^(-1)
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
d_0 = 400 * 10 ** (-5)  # m # 500*10**(-5)
Ne_0 = 10 ** (20)  # cm^-3 # 10 ** (21)
t_0 = 10 ** (14)  # 1/tau # 10*(15)
eps_Inf_0 = 3  # 5


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


# ==========================================================#
#                  CALCULATING T AND R                     #
# ==========================================================#
start1 = time.time()
eps2 = 10 ** (-5)  # available error


def mmt(x, Ne, eps_Inf, t, dm):  # x - size w, other parameters membrane
    Tm = []  # list of T
    Rm = []  # list of R
    for m in range(0, x):
        wp = w_p(Ne, eps_Inf)  # omega plasmon
        eps_ = eps(eps_Inf, wp, t)
        nm = n_m(eps_[m])
        km = k_m(eps_[m])
        D0 = (1 / 2) * (1 / nm) * np.array([[nm + n_vac[m], nm - n_vac[m]],
                                            [nm - n_vac[m], nm + n_vac[m]]])
        # 1
        P_m = np.array([[np.exp(i * fi(w[m], n_(nm, km), dm)), 0],
                        [0, np.exp(-i * fi(w[m], n_(nm, km), dm))]])
        # 1-2
        D1 = (1 / (2 * n_(n_K108[m], k_K108[m]))) * np.matrix(
            [[nm + n_(n_K108[m], k_K108[m]), n_(n_K108[m], k_K108[m]) - nm],
             [n_(n_K108[m], k_K108[m]) - nm, nm + n_(n_K108[m], k_K108[m])]])
        # 2
        P_K108 = np.array([[np.exp(i * fi(w[m], n_(n_K108[m], k_K108[m]), d_K108)), 0],
                           [0, np.exp(-i * fi(w[m], n_(n_K108[m], k_K108[m]), d_K108))]])
        # 2-3
        D2 = (1 / (2 * n_vac[m])) * np.array(
            [[n_vac[m] + n_(n_K108[m], k_K108[m]), n_vac[m] - n_(n_K108[m], k_K108[m])],
             [(n_vac[m] - n_(n_K108[m], k_K108[m])), n_vac[m] + n_(n_K108[m], k_K108[m])]])
        M = D2 @ P_K108 @ D1 @ P_m @ D0
        T = abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1])
        Tm.append(T)
        R = abs(M[1, 0] / M[1, 1]) ** (2)
        Rm.append(R)
    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack([[T1], [R1]])
    return eqn1


# ==========================================================#
#                  CALCULATE RMS DEVIATION                 #
# ==========================================================#
S = mmt(indx2, Ne_0, eps_Inf_0, t_0, d_0) - full_exp
print(S, S.shape)
S_T_0 = np.std(S[0, :])
print(S_T_0)
S_R_0 = np.std(S[1, :])
print(S_R_0)


# ==========================================================#
#                  FIND OPTIMAL PARAMETERS                 #
# ==========================================================#

def minimization(d_0, Ne_0, t_0, eps_Inf_0, eps2):
    S = mmt(indx2, Ne_0, eps_Inf_0, t_0, d_0) - full_exp
    print(S, S.shape)
    S_T_0 = np.std(S[0, :])
    print(S_T_0)
    S_R_0 = np.std(S[1, :])
    print(S_R_0)
    while S_T_0 and S_R_0 > eps2:
        if S_T_0 - eps2 > eps2 * 10 ** (4):
            d_0 += random.uniform(0, 10)
            Ne_0 += random.uniform(0, 10)
            t_0 += random.uniform(0, 10)
            eps_Inf_0 += random.uniform(0, 0.01)
            if d_0 > 500 * 10 ** (-5):
                d_0 -= random.uniform(10, 15)
            if Ne_0 > 10 ** (21):
                d_0 -= random.uniform(10, 15)
            if t_0 > 10 ** (15):
                t_0 -= random.uniform(10, 15)
            if eps_Inf_0 > 5:
                eps_Inf_0 -= random.uniform(0.01, 0.02)
        else:
            d_0 += random.uniform(0, 100)
            Ne_0 += random.uniform(0, 100)
            t_0 += random.uniform(0, 100)
            eps_Inf_0 += random.uniform(0, 0.1)
            if d_0 > 500 * 10 ** (-5):
                d_0 -= random.uniform(100, 120)
            if Ne_0 > 10 ** (21):
                d_0 -= random.uniform(100, 120)
            if t_0 > 10 ** (15):
                t_0 -= random.uniform(100, 120)
            if eps_Inf_0 > 5:
                eps_Inf_0 -= random.uniform(100, 120)
    S = mmt(indx2, Ne_0, eps_Inf_0, t_0, d_0) - full_exp
    S_T_0 = np.std(S[0, :])
    S_R_0 = np.std(S[1, :])
    return S_T_0, S_R_0, d_0, Ne_0, t_0, eps_Inf_0
print(minimization(d_0, Ne_0, t_0, eps_Inf_0, eps2))

end1 = time.time()
print(end1 - start1)

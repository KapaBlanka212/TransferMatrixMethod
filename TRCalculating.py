import time
import MaterialConstant
import numpy as np
from numpy import *
import scipy as sp
#from sympy import Symbol
from scipy.optimize import minimize
import scipy.integrate as integrate
import matplotlib.pyplot as plt

const = MaterialConstant.MaterialConstant()
# ==========================================================#
#                     GLOBAL CONSTANT                       #
# ==========================================================#
m0 = 9.1 * 10 ** (-28)  # electron mass g
me = 0.35 * m0  # electron mass in thin film
e0 = 4.8 * 10 ** (-10)  #
c = const.lightspeed  # cm/s
l = const.short_wavelenght * 10 ** (-4)  # cm
print(l)
l_full = const.full_wavelength * 10 ** (-4) # cm
w = 2 * pi * c / (l)  # sec^(-1)
w_l = 2 * pi * c / 0.2 # 1/s
w_full = 2 * pi * c / (l_full)  # sec^(-1)
i = 1j  # imag 1
# ==========================================================#
#                    CHOOSING  THIN FILM                    #
# ==========================================================#
th = int(input('Input the number of thin film '))
# ==========================================================#
#                     K108 PARAMETERS                       #
# ==========================================================#
d_K108 = const.d_K108  # cm
n_K108 = const.short_n # for short wavelength (0.4 to 4 mcm)
k_K108 = const.short_k # for short wavelength (0.4 to 4 mcm)
n_th_w = 2.5  # # n_k108 for l = 2000 mcm
k_th_w = 2.49 * 0.2 / (4 * pi)  # k_K108 l = 2000 mcm
n_K108_full = const.n_K108 # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
k_K108_full = const.k_K108 # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
# ==========================================================#
#                     VACUUM PARAMETERS                     #
# ==========================================================#
n_vac = const.n_vac  # VACUUM
k_vac = const.k_vac  # 0 == 0
# ==========================================================#
#                 EXPERIMENTAL VALUE T and R                #
# ==========================================================#
def TR_exp_2000(T_exp_long,R_exp_long):
    T_exp_2000 = T_exp_long
    # 4 * 10 ** (-4)  # # T for long wavelength (l > 300 mcm)
    R_exp_2000 = R_exp_long
    # 0.95  # # R for long wavelength (l > 300 mcm)
    return T_exp_2000,R_exp_2000

def TR_exp_short(T_exp,R_exp):
    T_exp = np.array([T_exp])  # T for short wavelength (0.4 to 4 mcm)
    R_exp = np.array([R_exp])  # R for short wavelength (0.4 to 4 mcm)
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])  # T and R matrix
    return full_exp, indx

if th == 6:
    full_exp = TR_exp_short(const.T6_short,const.R6_short)[0]
    print(full_exp)
    #TR_exp_2000(0.0004,0.95)
    indx = TR_exp_short(const.T6_short,const.R6_short)[1]  # need for mmt(x)
    T_exp_2000 = TR_exp_2000(0.0004,0.95)[0]
    R_exp_2000 = TR_exp_2000(0.0004,0.95)[1]

elif th == 5:
    full_exp = TR_exp_short(const.T5_short, const.R5_short)[0]
    print(full_exp)
    TR_exp_2000(0.0025, 0.93)
    indx = TR_exp_short(const.T5_short, const.R5_short)[1]  # need for mmt(x)

elif th == 4:
    full_exp = TR_exp_short(const.T4_short, const.R4_short)[0]
    print(full_exp)
    TR_exp_2000(0.0044, 0.89)
    indx = TR_exp_short(const.T6_short, const.R6_short)[1]  # need for mmt(x)

elif th == 3:
    full_exp = TR_exp_short(const.T3_short, const.R3_short)[0]
    print(full_exp)
    TR_exp_2000(0.0082, 0.87)
    indx = TR_exp_short(const.T3_short, const.R3_short)[1]  # need for mmt(x)

elif th == 2:
    full_exp = TR_exp_short(const.T2_short, const.R2_short)[0]
    print(full_exp)
    TR_exp_2000(0.011, 0.82)
    indx = TR_exp_short(const.T2_short, const.R2_short)[1]  # need for mmt(x)

    #np.savetxt('TR_exp_2', full_exp)
elif th == 1:
    full_exp = TR_exp_short(const.T1_short, const.R1_short)[0]
    print(full_exp)
    TR_exp_2000(0.016, 0.78)
    indx = TR_exp_short(const.T1_short, const.R1_short)[1]  # need for mmt(x)


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
#                      THEORY DRUDE                         #
# ==========================================================#


def eps(eps_inf, wp, t, w):  # t = 1/tau
    wp2 = wp ** 2
    w_ = w*(w+i*t)
    eps_return = eps_inf * (1.0 - wp2 / w_)
    return eps_return


def w_p(ne, eps_inf):
    w_p_out = np.sqrt((4 * pi * ne * e0 ** (2)) / (eps_inf * me))
    return w_p_out


def n_m(eps):
#    n_m_out = (1/np.sqrt(2)) * np.sqrt(np.sqrt((np.real(eps)) ** 2 + (np.imag(eps)) ** 2) + np.real(eps))
    n_m_out = np.real(np.sqrt(eps))
    return n_m_out


def k_m(eps):
#    k_m_out = (1/np.sqrt(2)) * np.sqrt(np.sqrt((np.real(eps)) ** 2 + (np.imag(eps)) ** 2) - np.real(eps))
    k_m_out = np.imag(np.sqrt(eps))
    return k_m_out


# ==========================================================#
#                   CALCULATING T AND R                     #
# ==========================================================#
start1 = time.time()

def P(fi):
    P_out = np.array([[np.exp(i * fi), 0],
                      [0, np.exp(-i * fi)]], dtype = object)
    return P_out

def D(n1,n2):
    A = 1 / (2*n2)
    D_out = A * np.array([[n1 + n2, n2 - n1],
                          [n2 - n1, n1 + n2]])
    return D_out

def x(ne_f, eps_inf_f, t_f, d_f):
    x_out = np.array([ne_f, eps_inf_f, t_f, d_f]).transpose()
    return x_out

z = indx
# x[0] - Ne
# x[1] - epsInf
# x[2] - 1/tau
# x[3] - d


def mmt_short(x):
#    M = [] # list transfer matrix
    Tm = []  # list of T
    Rm = []  # list of R
    a = 10
    wp = w_p(x[0], x[1])  # omega plasmon
    eps_ = eps(x[1], wp, x[2], w)
    nm = n_m(eps_)
    print('преломление плёнки',nm)
    km = k_m(eps_)
    for m in range(0, z):
        def matrix(n):
            delta = np.linspace(0, 2 * pi, n)
            # 0/1
            D1 = D(n_vac,nm[m])
            # 1
            P1 = P(fi(w[m],n_(nm[m], km[m]), x[3]))
            # 1/2
            D2 = D(nm[m],n_(n_K108[m], k_K108[m]))
            # 2
            P2 = P(fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))
            # 2/3
            D3 = D(n_(n_K108[m], k_K108[m]),n_vac)
            # transfer matrix
            M = D3 @ P2 @ D2 @ P1 @ D1
            return M
        #start = time.time()
        M1 = matrix(a)
        M2 = matrix(a*2)
        T_ = (abs(M2[0, 0] - ((M2[0, 1] * M2[1, 0]) / M2[1, 1]))) ** (2)
        R_ = (abs(M2[1, 0] / M2[1, 1])) ** (2)
        T = (abs(M1[0, 0] - ((M1[0, 1] * M1[1, 0]) / M1[1, 1]))) ** (2)
        R = (abs(M1[1, 0] / M1[1, 1])) ** (2)
        delta1 = np.linspace(0, 2 * pi, a)
        delta2 = np.linspace(0, 2 * pi, 2*a)
        t_ = integrate.simpson(T_, delta2) * (1 / (2 * pi))
        r_ = integrate.simpson(R_, delta2) * (1 / (2 * pi))
        t = integrate.simpson(T, delta1) * (1 / (2 * pi))
        r = integrate.simpson(R, delta1) * (1 / (2 * pi))
        err = abs(r - r_) / (abs(r_))
        if err > 10 ** (-5):
            a = 2*a
        #end = time.time()
        #tim = end - start
        Rm.append(r_)

        Tm.append(t_)
    #print(tim)
    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack((T1, R1))
    #print(eqn1)
    return eqn1


def mmt_2000(x):
    wp = w_p(x[0], x[1])  # omega plasmon
    eps_ = eps(x[1], wp, x[2], 0.2)
    nm = n_m(eps_)
    km = k_m(eps_)
    def matrix(delta1):
        # 0/1
        D1 = D(n_vac,nm)
        # 1
        P1 = P(fi(0.2,n_(nm, km), x[3]))
        # 1/2
        D2 = D(nm,n_(n_th_w, k_th_w))
        # 2
        P2 = P(fi_(0.2, n_(n_th_w, k_th_w), d_K108, delta1))
        # 2/3
        D3 = D(n_(n_th_w, k_th_w),n_vac)
        # transfer matrix
        M_out = D3 @ P2 @ D2 @ P1 @ D1
        return M_out


    def integrateT(delta1):
        M = matrix(delta1)
        T_d = (abs(M[0, 0] - ((M[0, 1] * M[1, 0]) / M[1, 1]))) ** (2)
        return T_d

    def integrateR(delta1):
        M = matrix(delta1)
        R_d = (abs(M[1, 0] / M[1, 1])) ** (2)
        return R_d

    T2 = integrate.quad(integrateT, 0, 2 * pi)
    T = (1 / (2 * pi)) * T2[0]
    R2 = integrate.quad(integrateR, 0, 2 * pi)
    R = (1 / (2 * pi)) * R2[0]
    eqn1 = np.vstack((T, R))
    return eqn1


# ==========================================================#
#           ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#
def Bounds(d_l,d_r,Ne_l,Ne_r,t_l,t_r,eps_l,eps_r):
    d_0_left = d_l  # cm
    d_0_right = d_r  # cm
    Ne_0_left = Ne_l  # cm^-3
    Ne_0_right = Ne_r  # cm^-3
    t_0_left = t_l  # 1/tau
    t_0_right = t_r  # 1/tau
    eps_inf_0_left = eps_l
    eps_inf_0_right = eps_r
    bnds = np.array([[Ne_0_left,Ne_0_right],
                     [eps_inf_0_left,eps_inf_0_right],
                     [t_0_left,t_0_right],
                     [d_0_left,d_0_right]])
    return bnds

if th == 6:
    bnds = Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
                  10 ** 20, 10 ** 21,
                  10 ** 14, 10 ** 15,
                  3.5, 5.0)

elif th == 5:
    Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
           10 ** 20, 10 ** 21,
           10 ** 14, 10 ** 15,
           3.5, 5.0)

elif th == 4:
    Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
           10 ** 20, 10 ** 21,
           10 ** 14, 10 ** 15,
           3.5, 5.0)

elif th == 3:
    Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
           10 ** 20, 10 ** 21,
           10 ** 14, 10 ** 15,
           3.5, 5.0)

elif th == 2:
    Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
           10 ** 20, 10 ** 21,
           10 ** 13, 10 ** 15,
           3.5, 5.0)

elif th == 1:
    Bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
           10 ** 20, 10 ** 21,
           10 ** 14, 10 ** 15,
           3.5, 5.0)

# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#
# TODO: find optimal parameters d,Ne,t,eps_Inf : in progress

alfa = 1
beta = 1
gamma = 1

def func(x): # target function
    #r = mmt_2000(x)
    #R_th_2000 = r[1,:]
    #T_th_2000 = r[0,:]
    s = mmt_short(x) - full_exp
    s1 = s[0,:]
    s2 = s[1,:]
    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    s1_s = sum1/indx
    s2_s = sum2/indx
    fun = gamma * np.sqrt((1/indx) * (np.sum((s1 - s1_s) ** 2) + np.sum((s2 - s2_s) ** 2)))# + \
          #alfa * (abs(R_exp_2000 - R_th_2000) / R_exp_2000) + beta * (abs(T_exp_2000 - T_th_2000) / T_exp_2000)
    print(fun,x)
    return fun



# zero approximation
x0 = np.array([1.0e+21,
               3.91,
               1.0e+14,
               4.200e-05]).transpose()
# bound for Nelder-Mead method


ans = sp.optimize.minimize(func,x0,method = 'Nelder-Mead',bounds=bnds,
                           options={'disp':True,'return_all': None,'adaptive': True,
                                    'maxiter':None})
print(ans)
# save ans
x_res = ans.x
# calculate for all wavelength
#indx2 = n_K108_full.size
#z = indx2
l1 = l * 10**(4)
TR = np.array(mmt_short(x_res)).transpose()
if th == 6:
    np.save('result\ TR6', TR)
    np.save('result\ res', x_res)
    np.savetxt('result\ TR6',TR)
    np.savetxt('result\ res',x_res)
elif th ==5:
    np.save('result\ TR5', TR)
    np.save('result\ res', x_res)
    np.savetxt('result\ TR5', TR)
    np.savetxt('result\ res', x_res)
elif th == 4:
    np.save('result\TR4', TR)
    np.save('result\ res', x_res)
    np.savetxt('result\ TR4', TR)
    np.savetxt('result\ res', x_res)
elif th == 3:
    np.save('result\ TR3', TR)
    np.save('result\ res', x_res)
    np.savetxt('result\ res', x_res)
    np.savetxt('result\ TR3',TR)
elif th == 2:
    np.save('result\ TR2', TR)
    np.save('result\ res', x_res)
    np.savetxt('result\ TR2',TR)
    np.savetxt('result\ res', x_res)
elif th == 1:
    np.save('TR1', TR)
    np.save('res', x_res)
    np.savetxt('TR1',TR)
    np.savetxt('res', x_res)
np.save('L',l1)
np.savetxt('L',l1)
end1 = time.time()
print(end1 - start1)

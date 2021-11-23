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
me = 0.35 * m0  # electron mass in thin membrane
e0 = 4.8 * 10 ** (-10)  #
c = const.lightspeed * 10 ** (2)   # cm/c
l = const.short_wavelenght * 10 ** (-4)  # cm
l_full = const.full_wavelength * 10 ** (-4) # cm
w = 2 * pi * c / (l)  # sec^(-1)
w_full = 2 * pi * c / (l_full)  # sec^(-1)
i = complex(0,1)  # imag 1
# ==========================================================#
#                    CHOOSING  THIN FILM                    #
# ==========================================================#
th = int(input('Input the number of thin film '))
# ==========================================================#
#                     K108 PARAMETERS                       #
# ==========================================================#
d_K108 = const.d_K108  # cm
n_K108 = const.short_n # for short wavelength (0.4 to 4 mcm)
k_K108 = (const.short_alfa) * (l / (4 * pi)) # for short wavelength (0.4 to 4 mcm)
n_th_w = 2.5  # # n_k108 for l = 2000 mcm
k_th_w = 2.49 * 0.2 / (4 * pi)  # k_K108 l = 2000 mcm
n_K108_full = const.n_K108 # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
k_K108_full = (const.alfa_K108) * (l_full / (4 * pi)) # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
# ==========================================================#
#                     VACUUM PARAMETERS                     #
# ==========================================================#
n_vac = const.n_vac  # VACUUM
k_vac = const.alfa_vac  # 0 == 0
# ==========================================================#
#                 EXPERIMENTAL VALUE T and R                #
# ==========================================================#

w_l = 2*pi*c/0.2 # 1/sec
if th == 6:
    T_exp_2000 = 4*10**(-4) # # T for long wavelength (l > 300 mcm)
    R_exp_2000 = 0.95 # # R for long wavelength (l > 300 mcm)
    T_exp = const.T6_short # T for short wavelength (0.4 to 4 mcm)
    R_exp = const.R6_short # R for short wavelength (0.4 to 4 mcm)
    indx = R_exp.size # need for mmt(x)
    T_exp_full = const.T_exp_6
    R_exp_full = const.R_exp_6
    full_exp = np.vstack([T_exp, R_exp])# T and R matrix
    np.savetxt('TR_exp',full_exp)

elif th == 5:
    T_exp = const.T_exp_5
    R_exp = const.R_exp_5
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])
elif th == 4:
    T_exp = const.T_exp_4
    R_exp = const.R_exp_4
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])
elif th == 3:
    T_exp = const.T_exp_3
    R_exp = const.R_exp_3
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])
elif th == 2:
    T_exp = const.T_exp_2
    R_exp = const.R_exp_2
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])
elif th == 1:
    T_exp = const.T_exp_1
    R_exp = const.R_exp_1
    indx = R_exp.size
    full_exp = np.vstack([T_exp, R_exp])



# ==========================================================#
#                      THEORY DRUDE                         #
# ==========================================================#


def eps(eps_inf, wp, t, w):  # t = 1/tau
    eps_return = eps_inf * (1.0 - ((wp ** 2) / (w * (w + i * t))))
    return eps_return


def w_p(ne, eps_inf):
    w_p_out = np.sqrt((4 * pi * ne * e0 ** (2)) / (eps_inf * me))
    return w_p_out


def n_m(eps):
    n_m_return = (1/np.sqrt(2))*np.sqrt(np.sqrt((np.real(eps))**2+np.imag(eps)**2)+np.real(eps))
    return n_m_return


def k_m(eps):
    k_m_return = (1/np.sqrt(2))*np.sqrt(np.sqrt((np.real(eps))**2+np.imag(eps)**2)-np.real(eps))
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


def x(ne_f,eps_inf_f, t_f, d_f):
    x_out = np.array([ne_f,eps_inf_f,t_f,d_f]).transpose()
    return x_out


z = indx
# x[0] - Ne
# x[1] - epsInf
# x[2] - 1/tau
# x[3] - d


def mmt_short(x):
    Tm = []  # list of T
    Rm = []  # list of R
    for m in range(0, z):
        def matrix(delta):
            wp = w_p(x[0], x[1])  # omega plasmon
            eps_ = eps(x[1], wp, x[2], w[m])
            nm = n_m(eps_)
            km = k_m(eps_)
            # 0/1
            D0 = (1 / (2 * nm)) * np.array([[nm + n_vac, nm - n_vac],
                                            [nm - n_vac, nm + n_vac]])
            # 1
            P_m = np.array([[np.exp(i * fi(w[m], n_(nm, km), x[3])), 0],
                            [0, np.exp(-i * fi(w[m], n_(nm, km), x[3]))]])
            # 1/2
            D1 = (1 / (2 * n_(n_K108[m], k_K108[m]))) * np.array([
                [nm + n_(n_K108[m], k_K108[m]), n_(n_K108[m], k_K108[m]) - nm],
                [n_(n_K108[m], k_K108[m]) - nm, nm + n_(n_K108[m], k_K108[m])]])

            # 2
            def P_K108(delta):
                P_K108_out = np.array([[np.exp(i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta)), 0],
                                       [0, np.exp(-i * fi_(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))]])
                return P_K108_out

            # 2/3
            D2 = (1 / (2 * n_vac)) * np.array([[n_vac + n_(n_K108[m], k_K108[m]), n_vac - n_(n_K108[m], k_K108[m])],
                                               [(n_vac - n_(n_K108[m], k_K108[m])), n_vac + n_(n_K108[m], k_K108[m])]])
            M_out = D2 @ P_K108(delta) @ D1 @ P_m @ D0
            return M_out


        def integrateT(delta):
            M = matrix(delta)
            T_d = abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1]) ** (2)
            return T_d

        T2 = integrate.quad(integrateT, 0, 2 * pi)
        T = (1 / (2 * pi)) * T2[0]
        Tm.append(T)

        def integrateR(delta):
            M = matrix(delta)
            R_d = abs(M[1, 0] / M[1, 1]) ** (2)
            return R_d


        R2 = integrate.quad(integrateR, 0, 2 * pi)
        R = (1 / (2 * pi)) * R2[0]
        Rm.append(R)
    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack([[T1], [R1]])
    #print(eqn1)
    return eqn1


def mmt_long(x):
    def matrix(delta):
        wp = w_p(x[0], x[1])  # omega plasmon
        eps_ = eps(x[1], wp, x[2], w_l)
        nm = n_m(eps_)
        km = k_m(eps_)
        D0 = (1 / 2) * (1 / nm) * np.array([[nm + n_vac, nm - n_vac],
                                            [nm - n_vac, nm + n_vac]])
        # 1
        P_m = np.array([[np.exp(i * fi(w_l, n_(nm, km), x[3])), 0],
                        [0, np.exp(-i * fi(w_l, n_(nm, km), x[3]))]])
        # 1-2
        D1 = (1 / (2 * n_(n_th_w, k_th_w))) * np.array([[nm + n_(n_th_w, k_th_w), n_(n_th_w, k_th_w - nm)],
                                                        [n_(n_th_w, k_th_w) - nm, nm + n_(n_th_w, k_th_w)]])

        # 2

        def P_K108(delta):
            P_K108_out = np.array([[np.exp(i * fi_(w_l, n_(n_th_w, k_th_w), d_K108, delta)), 0],
                                   [0, np.exp(-i * fi_(w_l, n_(n_th_w, k_th_w), d_K108, delta))]])
            return P_K108_out

        # 2-3
        D2 = (1 / (2 * n_vac)) * np.array([[n_vac + n_(n_th_w, k_th_w), n_vac - n_(n_th_w, k_th_w)],
                                           [(n_vac - n_(n_th_w, k_th_w)), n_vac + n_(n_th_w, k_th_w)]])
        M_out = D2 @ P_K108(delta) @ D1 @ P_m @ D0
        return M_out

    def integrateT(delta):
        M = matrix(delta)
        T_d = abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1]) ** (2)
        return T_d

    T1 = integrate.quad(integrateT, 0, 2 * pi)
    T = (1 / (2 * pi)) * T1[0]

    def integrateR(delta):
        M = matrix(delta)
        R_d = abs(M[1, 0] / M[1, 1]) ** (2)
        return R_d

    R1 = integrate.quad(integrateR, 0, 2 * pi)
    R = (1 / (2 * pi)) * R1[0]
    eqn2 = np.vstack([[T], [R]])
    return eqn2


def mmt_full(x):
    Tm = []  # list of T
    Rm = []  # list of R
    for m in range(0, z):
        def matrix(delta):
            wp = w_p(x[0], x[1])  # omega plasmon
            eps_ = eps(x[1], wp, x[2], w_full[m])
            nm = n_m(eps_)
            km = k_m(eps_)
            D0 = (1 / 2) * (1 / nm) * np.array([[nm + n_vac, nm - n_vac],
                                                [nm - n_vac, nm + n_vac]])
            # 1
            P_m = np.array([[np.exp(i * fi(w_full[m], n_(nm, km), x[3])), 0],
                            [0, np.exp(-i * fi(w_full[m], n_(nm, km), x[3]))]])
            # 1-2
            D1 = (1 / (2 * n_(n_K108_full[m], k_K108_full[m]))) * np.array(
                [[nm + n_(n_K108_full[m], k_K108_full[m]), n_(n_K108_full[m], k_K108_full[m]) - nm],
                 [n_(n_K108_full[m], k_K108_full[m]) - nm, nm + n_(n_K108_full[m], k_K108_full[m])]])

            # 2
            def P_K108(delta):
                P_K108_out = np.array(
                    [[np.exp(i * fi_(w_full[m], n_(n_K108_full[m], k_K108_full[m]), d_K108, delta)), 0],
                     [0, np.exp(-i * fi_(w_full[m], n_(n_K108_full[m], k_K108_full[m]), d_K108, delta))]])
                return P_K108_out

            # 2-3
            D2 = (1 / (2 * n_vac)) * np.array([
                [n_vac + n_(n_K108_full[m], k_K108_full[m]), n_vac - n_(n_K108_full[m], k_K108_full[m])],
                [(n_vac - n_(n_K108_full[m], k_K108_full[m])), n_vac + n_(n_K108_full[m], k_K108_full[m])]])
            M_out = D2 @ P_K108(delta) @ D1 @ P_m @ D0
            return M_out


        def integrateT(delta):
            M = matrix(delta)
            T_d = abs(M[0, 0] - (M[0, 1] * M[1, 0]) / M[1, 1]) ** (2)
            return T_d


        T2 = integrate.quad(integrateT, 0, 2 * pi)
        T = (1 / (2 * pi)) * T2[0]
        Tm.append(T)


        def integrateR(delta):
            M = matrix(delta)
            R_d = abs(M[1, 0] / M[1, 1]) ** (2)
            return R_d


        R2 = integrate.quad(integrateR, 0, 2 * pi)
        R = (1 / (2 * pi)) * R2[0]
        Rm.append(R)
    T1 = np.array(Tm)
    R1 = np.array(Rm)
    eqn1 = np.vstack([[T1], [R1]])
    return eqn1


# ==========================================================#
#           ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#
d_0_left = 400 * 10 ** (-7)  # cm
d_0_right = 600 * 10 ** (-7)  # cm
Ne_0_left = 10 ** (20)  # cm^-3
Ne_0_right = 10 ** (21)  # cm^-3
t_0_left = 10 ** (13)  # 1/tau
t_0_right = 10 ** (14)  # 1/tau
eps_inf_0_left = 3.0
eps_inf_0_right =  5.0
# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#
# TODO: find optimal parameters d,Ne,t,eps_Inf : in progress

alfa = 1
beta = 1
gamma = 1

def func(x): # target function
    th = mmt_long(x)
    R_th_2000 = th[1,:]
    T_th_2000 = th[0,:]
    s = mmt_short(x) - full_exp
    s1 = s[0,:]
    s2 = s[1,:]
    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    s1_s = sum1/indx
    s2_s = sum2/indx
    fun = gamma*np.sqrt((1/indx)*(np.sum((s1-s1_s)**2)+np.sum((s2-s2_s)**2)))#+\
          #alfa*(abs(R_exp_2000-R_th_2000)/R_exp_2000)+beta*(abs(T_exp_2000-T_th_2000)/T_exp_2000)
    print(fun,x)
    return fun



# zero approximation
x0 = np.array([0.79 * 10**(21),
               4.0,
               1*10**(14),
               500*10**(-7)]).transpose()
# bound for Nelder-Mead method
bnds = ((Ne_0_left,Ne_0_right),
        (eps_inf_0_left,eps_inf_0_right),
        (t_0_left,t_0_right),
        (d_0_left,d_0_right))

ans = sp.optimize.minimize(func,x0,method = 'Nelder-Mead',bounds=bnds,tol = 0.1,
                           options={'disp':True,'return_all': None,'adaptive': None,
                                    'maxiter':None})
print(ans)
# save ans
x_res = ans.x
# calculate for all wavelength
indx2 = n_K108_full.size
z = indx2
l1 = l * 10**(4)
TR = np.array(mmt_full(x_res)).transpose()
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

import time

import numpy as np
from numpy import pi
import scipy as sp
from scipy.optimize import minimize
import scipy.integrate as integrate

import MaterialConstant

constant = MaterialConstant.MaterialConstant()
# ==========================================================#
#                     GLOBAL CONSTANT                       #
# ==========================================================#
m0 = 9.1 * 10 ** (-28)  # electron mass g
me = 0.35 * m0  # electron mass in thin film
e0 = 4.8 * 10 ** (-10)  # SGS
c = constant.lightspeed  # cm/s
len = constant.short_wavelenght * 10 ** (-4)  # cm
len_full = constant.full_wavelength * 10 ** (-4)  # cm
w = 2 * pi * c / len  # sec^(-1)
w_l = 2 * pi * c / 0.2  # 1/s
w_full = 2 * pi * c / len_full  # sec^(-1)
i = 1j  # image 1
# ==========================================================#
#                    CHOOSING  THIN FILM                    #
# ==========================================================#
th = int(input('Input the number of thin film '))
# ==========================================================#
#                     K108 PARAMETERS                       #
# ==========================================================#
d_K108 = constant.d_K108  # cm
n_K108 = constant.short_n  # for short wavelength (0.4 to 4 mcm)
k_K108 = constant.short_k  # for short wavelength (0.4 to 4 mcm)
n_th_w = 2.5  # n_k108 for l = 2000 mcm
k_th_w = 2.49 * 0.2 / (4 * pi)  # k_K108 l = 2000 mcm
n_K108_full = constant.n_K108  # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
k_K108_full = constant.k_K108  # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
# ==========================================================#
#                     VACUUM PARAMETERS                     #
# ==========================================================#
n_vac = constant.n_vac  # VACUUM
k_vac = constant.k_vac  # 0 == 0
# ==========================================================#
#                 EXPERIMENTAL VALUE T and R                #
# ==========================================================#


def tr_exp_2000(t_exp_long, r_exp_long):
    tra_exp_2000 = t_exp_long
    ref_exp_2000 = r_exp_long
    return tra_exp_2000, ref_exp_2000


def tr_exp_short(t_exp, r_exp):
    tr_exp = t_exp  # T for short wavelength (0.4 to 4 mcm)
    ref_exp = r_exp  # R for short wavelength (0.4 to 4 mcm)
    ind = r_exp.size
    full_experiment = np.vstack([tr_exp, ref_exp])  # T and R matrix
    return full_experiment, ind


if th == 6:
    full_exp = tr_exp_short(constant.T6_short, constant.R6_short)[0]
    np.savetxt('TR6_EXP', np.transpose(full_exp))
    index = tr_exp_short(constant.T6_short, constant.R6_short)[1]  # need for mmt(x)
    t_exp_2000 = tr_exp_2000(3.9e-4, 0.95)[0]
    r_exp_2000 = tr_exp_2000(3.9e-4, 0.95)[1]

elif th == 5:
    full_exp = tr_exp_short(constant.T5_short, constant.R5_short)[0]
    np.savetxt('TR5_EXP', np.transpose(full_exp))
    tr_exp_2000(0.0025, 0.93)
    index = TR_exp_short(constant.T5_short, constant.R5_short)[1]  # need for mmt(x)

elif th == 4:
    full_exp = tr_exp_short(constant.T4_short, constant.R4_short)[0]
    np.savetxt('TR4_EXP', np.transpose(full_exp))
    tr_exp_2000(0.0044, 0.89)
    index = tr_exp_short(constant.T6_short, constant.R6_short)[1]  # need for mmt(x)

elif th == 3:
    full_exp = tr_exp_short(constant.T3_short, constant.R3_short)[0]
    np.savetxt('TR3_EXP', np.transpose(full_exp))
    tr_exp_2000(0.0082, 0.87)
    index = tr_exp_short(constant.T3_short, constant.R3_short)[1]  # need for mmt(x)

elif th == 2:
    full_exp = tr_exp_short(constant.T2_short, constant.R2_short)[0]
    np.savetxt('TR2_EXP', np.transpose(full_exp))
    tr_exp_2000(0.011, 0.82)
    index = tr_exp_short(constant.T2_short, constant.R2_short)[1]  # need for mmt(x)

elif th == 1:
    full_exp = tr_exp_short(constant.T1_short, constant.R1_short)[0]
    np.savetxt('TR1_EXP', np.transpose(full_exp))
    tr_exp_2000(0.016, 0.78)
    index = tr_exp_short(constant.T1_short, constant.R1_short)[1]  # need for mmt(x)


# ==========================================================#
#      Complex refractive index and electric potential      #
# ==========================================================#


def n_(n, k):
    n_return = n + i * k
    return n_return


def fi(fr, n, d):
    a = fr / c
    fi_out = a * n * d
    return fi_out


def fi_k108(fr, n, d, delta):
    a = fr / c
    fi_out = a * n * d + delta/2
    return fi_out


# ==========================================================#
#                      THEORY DRUDE                         #
# ==========================================================#


def eps(eps_inf, wp, t, fr):  # t = 1/tau
    wp2 = wp ** 2
    w_ = fr*(fr+i*t)
    eps_return = eps_inf * (1.0 - wp2 / w_)
    return eps_return


def w_p(ne, eps_inf):
    mass_const = (4 * pi * e0 ** 2) / me
    w_p_out = np.sqrt(mass_const * ne / eps_inf)
    return w_p_out


def n_m(epsilon):
    n_m_out = (1/np.sqrt(2)) * np.sqrt(np.sqrt((np.real(epsilon)) ** 2 + (np.imag(epsilon)) ** 2) + np.real(epsilon))
    return n_m_out


def k_m(epsilon):
    k_m_out = (1/np.sqrt(2)) * np.sqrt(np.sqrt((np.real(epsilon)) ** 2 + (np.imag(epsilon)) ** 2) - np.real(epsilon))
    return k_m_out


# ==========================================================#
#                   CALCULATING T AND R                     #
# ==========================================================#
start1 = time.time()
z = index
CONST = (1 / (2 * pi))

def P(fi):
    p_out = np.array([[np.exp(i * fi), 0],
                      [0, np.exp(-i * fi)]], dtype=object)
    return p_out

def D(n1, n2):
    A = 1 / (2*n2)
    d_out = A * np.array([[n1 + n2, n2 - n1],
                          [n2 - n1, n1 + n2]])
    return d_out


def x(ne_f, eps_inf_f, t_f, d_f):
    x_out = np.array([ne_f,  # concentration
                      eps_inf_f,  # high frequency dielectric constant
                      t_f,  # electron relaxation time
                      d_f])  # film thickness
    return x_out


def tra_mat(mat):
    t_d = (abs(mat[0, 0] - ((mat[0, 1] * mat[1, 0]) / mat[1, 1]))) ** 2
    return t_d


def re_mat(mat):
    r_d = (abs(mat[1, 0] / mat[1, 1])) ** 2
    return r_d


def mmt_short(par):
    tm = []  # list of T
    rm = []  # list of R
    a = 10
    wp = w_p(par[0], par[1])  # omega plasmon
    eps_ = eps(par[1], wp, par[2], w)
    nm = np.array(n_m(eps_))
    km = np.array(k_m(eps_))

    def matrix(n):
        delta = np.linspace(-pi, pi, n)
        # 0/1 layer
        d1 = D(n_vac, n_(nm[m], km[m]))
        # 1 layer
        p1 = P(fi(w[m], n_(nm[m], km[m]), par[3]))
        # 1/2 layer
        d2 = D(n_(nm[m], km[m]), n_(n_K108[m], k_K108[m]))
        # 2 layer
        p2 = P(fi_k108(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))
        # 2/3 layer
        d3 = D(n_(n_K108[m], k_K108[m]), n_vac)
        # transfer matrix
        matrix_out = d3 @ p2 @ d2 @ p1 @ d1
        return matrix_out
    for m in range(0, z):
        mat1 = matrix(a)
        mat2 = matrix(a*2)
        transmittance2 = tra_mat(mat2)
        reflectance2 = re_mat(mat2)
        transmittance1 = tra_mat(mat1)
        reflectance1 = re_mat(mat1)
        delta1 = np.linspace(0, 2 * pi, a)
        delta2 = np.linspace(0, 2 * pi, 2*a)
        transmittance2_int = integrate.simpson(transmittance2, delta2) * CONST
        reflectance2_int = integrate.simpson(reflectance2, delta2) * CONST
        transmittance1_int = integrate.simpson(transmittance1, delta1) * CONST
        reflectance1_int = integrate.simpson(reflectance1, delta1) * CONST
        err_r = abs(reflectance1_int - reflectance2_int) / (abs(reflectance2_int))
        err_t = abs(transmittance1_int - transmittance2_int) / (abs(transmittance2_int))
        if err_r > 10 ** (-5):
            a = 2 * a
        if err_t > 10 ** (-5):
            a = 2 * a
        rm.append(reflectance2_int)
        tm.append(transmittance2_int)
    t1 = np.array(tm)
    r1 = np.array(rm)
    eqn1 = np.vstack((t1, r1))
    return eqn1


def mmt_2000(par):
    wp = w_p(par[0], par[1])  # omega plasmon
    eps_ = eps(par[1], wp, par[2], 0.2)
    nm = n_m(eps_)
    km = k_m(eps_)

    def matrix(delta1):
        # 0/1
        d1 = D(n_vac, n_(nm, km))
        # 1
        p1 = P(fi(0.2, n_(nm, km), par[3]))
        # 1/2
        d2 = D(n_(nm, km), n_(n_th_w, k_th_w))
        # 2
        p2 = P(fi_k108(0.2, n_(n_th_w, k_th_w), d_K108, delta1))
        # 2/3
        d3 = D(n_(n_th_w, k_th_w), n_vac)
        # transfer matrix
        m_out = d3 @ p2 @ d2 @ p1 @ d1
        return m_out

    def int_transmittance(delta1):
        m = matrix(delta1)
        transmittance_k108 = (abs(m[0, 0] - ((m[0, 1] * m[1, 0]) / m[1, 1]))) ** 2
        return transmittance_k108

    def int_reflectance(delta1):
        m = matrix(delta1)
        reflectance_k108 = (abs(m[1, 0] / m[1, 1])) ** 2
        return reflectance_k108

    t = integrate.quad(int_transmittance, -pi, pi)
    t2 = CONST * t[0]
    r = integrate.quad(int_reflectance, -pi, pi)
    r2 = CONST * r[0]
    eqn1 = np.vstack((t2, r2))
    return eqn1


def mmt_full(par):
    t_list_long = []  # list of T
    r_list_long = []  # list of R
    a = 10  # amount of dot on first integrate step
    wp = w_p(par[0], par[1])  # omega plasmon
    eps_ = eps(par[1], wp, par[2], w_full)
    nm = (n_m(eps_))
    km = (k_m(eps_))

    def matrix(n):
        delta = np.linspace(0, 2 * pi, n)
        # 0/1
        d1 = D(n_vac, n_(nm[m], km[m]))
        # 1
        p1 = P(fi(w_full[m], n_(nm[m], km[m]), par[3]))
        # 1/2
        d2 = D(n_(nm[m], km[m]), n_(n_K108_full[m], k_K108_full[m]))
        # 2
        p2 = P(fi_k108(w_full[m], n_(n_K108_full[m], k_K108_full[m]), d_K108, delta))
        # 2/3
        d3 = D(n_(n_K108_full[m], k_K108_full[m]), n_vac)
        # transfer matrix
        m_out = d3 @ p2 @ d2 @ p1 @ d1
        return m_out
    for m in range(0, z):
        # create transfer matrix
        mat1 = matrix(a)
        mat2 = matrix(a * 2)
        # transmittance
        transmittance1 = T(mat1)
        transmittance2 = T(mat2)
        # reflectance
        reflectance2 = R(mat2)
        reflectance1 = R(mat1)
        # integration bounds
        delta1 = np.linspace(0, 2 * pi, a)
        delta2 = np.linspace(0, 2 * pi, 2 * a)
        # transmittance with 2 * a dot for integrate
        transmittance_int_2a = integrate.simpson(transmittance2, delta2) * (1 / (2 * pi))
        # transmittance with a * a dot for integrate
        transmittance_int_1a = integrate.simpson(transmittance1, delta1) * (1 / (2 * pi))
        # reflectance with 2 * a dot
        reflectance_int_2a = integrate.simpson(reflectance2, delta2) * (1 / (2 * pi))
        # reflectance with a dof
        reflectance_int_1a = integrate.simpson(reflectance1, delta1) * (1 / (2 * pi))
        # need for stop increase number of dot a
        err_r = abs(reflectance_int_1a - reflectance_int_2a) / (abs(reflectance_int_1a))
        err_t = abs(transmittance_int_1a - transmittance_int_2a) / (abs(transmittance_int_1a))
        if err_r > 10 ** -5:  # need for controlling amount of error of numerical integrate
            a = 2 * a
        if err_t > 10 ** -5:
            a = 2 * a
        r_list_long.append(reflectance_int_2a)  # add point in massive
        t_list_long.append(transmittance_int_2a)  # add point in massive
    tr_matrix_long = np.array(tm)  # type list - > type matrix
    re_matrix_long = np.array(rm)
    eqn1 = np.vstack((tr_matrix_long, re_matrix_long))  # massive of transfer matrix
    eqn1 = np.transpose(eqn1)
    return eqn1


# ==========================================================#
#           ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#


def bounds(d_l, d_r, ne_l, ne_r, t_l, t_r, eps_l, eps_r):
    bound = np.array([[ne_l, ne_r],
                     [eps_l, eps_r],
                     [t_l, t_r],
                     [d_l, d_r]])
    return bound


if th == 6:
    bnd = bounds(4.0e-5, 4.5e-5,
                 9.5e20, 9.8e20,
                 8e13, 10.0e13,
                 4.0, 4.2)

elif th == 5:
    bnd = bounds(4.0e-5, 5.0e-5,
                 1.0e20, 1.0e21,
                 1.0e14, 1.0e15,
                 3.5, 5.0)

elif th == 4:
    bnd = bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
                 10 ** 20, 10 ** 21,
                 10 ** 14, 10 ** 15,
                 3.5, 5.0)

elif th == 3:
    bnd = bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
                 10 ** 20, 10 ** 21,
                 10 ** 14, 10 ** 15,
                 3.5, 5.0)

elif th == 2:
    bnd = bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
                 10 ** 20, 10 ** 21,
                 10 ** 13, 10 ** 15,
                 3.5, 5.0)

elif th == 1:
    bnd = bounds(4 * 10 ** (-5), 5 * 10 ** (-5),
                 10 ** 20, 10 ** 21,
                 10 ** 14, 10 ** 15,
                 3.5, 5.0)

# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#


def func(par):  # target function
    alfa = 1
    beta = 1
    gamma = 1
    r = mmt_2000(par)
    r_th_2000 = r[1, :]
    t_th_2000 = r[0, :]
    s = abs(mmt_short(par) - full_exp)
    s1 = s[0, :]
    s2 = s[1, :]
    short_wavelength = gamma * ((np.sqrt((1 / index) * (np.sum(s1 ** 2)))) + np.sqrt((1/index) * (np.sum(s2 ** 2))))
    long_wavelength_r = alfa * (abs(r_exp_2000 - r_th_2000) / abs(r_exp_2000))
    long_wavelength_t = beta * (abs(t_exp_2000 - t_th_2000) / abs(t_exp_2000))
    fun = short_wavelength + long_wavelength_r + long_wavelength_t
    print('Target function', float(fun), 'Parameters: ',
          'Ne :', par[0],
          'eps_inf: ', par[1],
          '1/tau: ', par[2],
          'd: ', par[3])
    return fun


# zero approximation
x0 = np.array([[9.68291676e+20, 4.05159480e+00, 8.45263626e+13, 4.42299975e-05]])
ans = sp.optimize.minimize(func, x0,
                           method='Nelder-Mead',
                           bounds=bnd,
                           options={'disp': True,
                                    'adaptive': True,
                                    'maxiter': None,
                                    'fatol' : 10 ** -4,
                                    'return_all': True})
print(ans)
# save ans
x_res = ans.x
# calculate for all wavelength
index_full = n_K108_full.size
z = index_full
l1 = len_full * 10 ** 4
TR = np.array(mmt_full(x_res))
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
np.save('L', l1)
np.savetxt('L', l1)
end1 = time.time()
print(end1 - start1)

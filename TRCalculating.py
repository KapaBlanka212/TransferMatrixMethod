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
length = constant.short_wavelenght * 10 ** (-4)  # cm
len_full = constant.full_wavelength * 10 ** (-4)  # cm
w = 2 * pi * c / length  # sec^(-1)
w_2000 = 2 * pi * c / 0.2  # 1/s
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
k_K108 = constant.short_alfa * c / (2 * w)  # for short wavelength (0.4 to 4 mcm)
n_th_w = 2.51  # n_k108 for l = 2000 mcm
k_th_w = 1.97 * 0.2 / (4 * pi)  # k_K108 l = 2000 mcm
n_K108_full = constant.n_K108  # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
k_K108_full = constant.alfa_K108 * c / (2 * w_full) # for l from 0.4 to 4 mcm and from 300 mcm to 5000 mcm
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
    index = tr_exp_short(constant.T6_short, constant.R6_short)[1]  # need for mmt(x)
    t_exp_2000 = tr_exp_2000(4.0e-4, 0.95)[0]
    r_exp_2000 = tr_exp_2000(4.0e-4, 0.95)[1]

elif th == 5:
    full_exp = tr_exp_short(constant.T5_short, constant.R5_short)[0]
    index = tr_exp_short(constant.T5_short, constant.R5_short)[1]  # need for mmt(x)
    t_exp_2000 = tr_exp_2000(0.0025, 0.93)[0]
    r_exp_2000 = tr_exp_2000(0.0025, 0.93)[1]


elif th == 4:
    full_exp = tr_exp_short(constant.T4_short, constant.R4_short)[0]
    t_exp_2000 = tr_exp_2000(0.0044, 0.89)[0]
    r_exp_2000 = tr_exp_2000(0.0044, 0.89)[1]
    index = tr_exp_short(constant.T6_short, constant.R6_short)[1]  # need for mmt(x)

elif th == 3:
    full_exp = tr_exp_short(constant.T3_short, constant.R3_short)[0]
    t_exp_2000 = tr_exp_2000(0.0082, 0.87)[0]
    r_exp_2000 = tr_exp_2000(0.0082, 0.87)[1]
    tr_exp_2000(0.0082, 0.87)
    index = tr_exp_short(constant.T3_short, constant.R3_short)[1]  # need for mmt(x)

elif th == 2:
    full_exp = tr_exp_short(constant.T2_short, constant.R2_short)[0]
    t_exp_2000 = tr_exp_2000(0.011, 0.82)[0]
    r_exp_2000 = tr_exp_2000(0.011, 0.82)[1]
    index = tr_exp_short(constant.T2_short, constant.R2_short)[1]  # need for mmt(x)

elif th == 1:
    full_exp = tr_exp_short(constant.T1_short, constant.R1_short)[0]
    t_exp_2000 = tr_exp_2000(0.016, 0.78)[0]
    r_exp_2000 = tr_exp_2000(0.016, 0.78)[1]
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
#                      THEORY DRUDE                        #
# ==========================================================#

# t == 1 / tau, [cm-1]

def eps(eps_inf, wp, t, fr):
    wp2 = wp ** 2
    w_ = fr * (fr + i * t)
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


def p_matrix(pot):
    p_out = np.array([[np.exp(i * pot), 0],
                      [0, np.exp(-i * pot)]], dtype=object)
    return p_out


def d_interface(n1, n2):
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
        d1 = d_interface(n_vac, n_(nm[m], km[m]))
        # 1 layer
        p1 = p_matrix(fi(w[m], n_(nm[m], km[m]), par[3]))
        # 1/2 layer
        d2 = d_interface(n_(nm[m], km[m]), n_(n_K108[m], k_K108[m]))
        # 2 layer
        p2 = p_matrix(fi_k108(w[m], n_(n_K108[m], k_K108[m]), d_K108, delta))
        # 2/3 layer
        d3 = d_interface(n_(n_K108[m], k_K108[m]), n_vac)
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




# ==========================================================#
#            ZERO APPROXIMATION OF PARAMETERS               #
# ==========================================================#

# this function create a bounds for Nelder - Mead minimize method
def bounds(d_l, d_r, ne_l, ne_r, t_l, t_r, eps_l, eps_r):
    bound = np.array([[ne_l, ne_r],
                      [eps_l, eps_r],
                      [t_l, t_r],
                      [d_l, d_r]])
    return bound


# for unequal ITO film we use unequal bounds
if th == 6:
    bnd = bounds(4.0e-5, 5.0e-5,
                 9.0e20, 1.0e21,
                 7e13, 1.5e14,
                 3.5, 4.5)

elif th == 5:
    bnd = bounds(1.0e-5, 5.0e-5,
                 1.0e20, 1.0e21,
                 1.0e14, 1.0e15,
                 3.5, 5.0)

elif th == 4:
    bnd = bounds(1 * 10 ** (-5), 1.5 * 10 ** (-5),
                 5.0e20, 1.0e21,
                 1.0e13, 1.0e15,
                 3.5, 5.0)

elif th == 3:
    bnd = bounds(1 * 10 ** (-5), 1.5 * 10 ** (-5),
                 5.0e20, 1.0e21,
                 1.0e13, 1.0e15,
                 3.5, 5.0)

elif th == 2:
    bnd = bounds(0.8 * 10 ** (-5), 1.5 * 10 ** (-5),
                 5.0e20, 1.0e21,
                 1.0e13, 1.0e15,
                 3.5, 5.0)

elif th == 1:
    bnd = bounds(0.5 * 10 ** (-5), 1.5 * 10 ** (-5),
                 5.0e20, 1.0e21,
                 1.0e13, 1.0e15,
                 3.5, 5.0)

# ==========================================================#
#                   FIND OPTIMAL PARAMETERS                 #
# ==========================================================#


def func(par):  # target function that use for describe of amount error between theoretical and experimental value
    s = abs(mmt_short(par) - full_exp)
    s1 = s[0, :]
    s2 = s[1, :]
    short_wavelength = np.sqrt((1 / index) * (np.sum(s1 ** 2))) + np.sqrt((1/index) * (np.sum(s2 ** 2)))
    fun = short_wavelength
    print('Parameters: ', '\n',
          'Ne :', par[0], ' ????^(-3)',
          'eps_inf: ', par[1], '\n',
          '1/tau: ', par[2], ' c',
          'd: ', par[3] * 10 ** 7, ' ????', '\n')
    # need for control  amount of error
    print('function: = ', fun)
    return fun


# zero approximation
x0 = np.array([8.95308842e+20, 4.02613498e+00, 1.20176453e+14, 1.82192422e-05])
# ans == result of minimization
ans = sp.optimize.minimize(func, x0,
                           method='Nelder-Mead',  # the method of minimization our function
                           bounds=bnd,
                           options={'disp': True,
                                    'adaptive': True,
                                    'maxiter': None,
                                    'fatol': 10 ** -4,
                                    'return_all': True})
print(ans)
# save ans
x_res = ans.x
# calculate for all wavelength
l1 = length * 10 ** 4
TR = np.transpose(np.array(mmt_short(x_res)))
# save result in txt file
if th == 6:
    np.save('result//TR6', TR)
    np.save('result//res6', x_res)
    np.savetxt('result//TR6', TR)
    np.savetxt('result//res6', x_res)
elif th == 5:
    np.save('result//TR5', TR)
    np.save('result//res5', x_res)
    np.savetxt('result//TR5', TR)
    np.savetxt('result//res5', x_res)
elif th == 4:
    np.save('result//TR4', TR)
    np.save('result//res4', x_res)
    np.savetxt('result//TR4', TR)
    np.savetxt('result//res4', x_res)
elif th == 3:
    np.save('result//TR3', TR)
    np.save('result//res3', x_res)
    np.savetxt('result//res3', x_res)
    np.savetxt('result//TR3', TR)
elif th == 2:
    np.save('result//TR2', TR)
    np.save('result//res2', x_res)
    np.savetxt('result//TR2', TR)
    np.savetxt('result//res2', x_res)
elif th == 1:
    np.save('result//TR1', TR)
    np.save('result//res1', x_res)
    np.savetxt('result//TR1', TR)
    np.savetxt('result//res1', x_res)
np.save('L', l1)
np.savetxt('L', l1)
end1 = time.time()
print('program work in min: ', (end1 - start1) / 60)

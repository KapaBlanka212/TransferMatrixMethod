import numpy as np
from numpy import *
import scipy as sp
from scipy import optimize
d = 0.3
T = np.load('ConstantNPY\T_K108_short.npy')
np.savetxt('T',T)
R = np.load('ConstantNPY\R_K108_short.npy')
L = np.load('ConstantNPY\wl.npy') * 10 ** (-4)
T_l = np.load('ConstantNPY\T_K108_long.npy')
R_l = np.load('ConstantNPY\R_K108_long.npy')
L_l = np.load('ConstantNPY\w_long.npy') * 10 ** (-4)
indx = L.size
indx1 = L_l.size
K = []
N = []
for m in range(0,indx):
    def eqn_r(x):
        eqn_r_out =(T[m])**2-(R[m])**2+3*R[m]*x-2*(x)**2-(R[m]/x)+1
        return eqn_r_out
    r = sp.optimize.newton(eqn_r,R[m],disp = True)
    alfa = -(1 / d) * np.log((R[m] - r) / (r * T[m]))
    k = (alfa * L[m]) / (4*pi)
    K.append(k)
    n = ((2 * r + 2) + np.sqrt((-2 * r - 2) ** 2 - 4 * (1-r) * (k ** 2 * (-(r-1)) - r + 1))) / (2 * ( 1 - r ))
    N.append(n)
anser = np.vstack([N,K])
N_l = []
K_l = []
for m in range(0,indx1):
    def eqn_r_l(x):
        eqn_r_out =(T_l[m])**2-(R_l[m])**2+3*R_l[m]*x-2*(x)**2-(R_l[m]/x)+1
        return eqn_r_out
    r_l = sp.optimize.newton(eqn_r_l , R_l[m] , disp = True)
    alfa_l = -(1 / d) * np.log((R_l[m] - r_l) / (r_l * T_l[m]))
    k_l = (alfa_l * L_l[m]) / (4*pi)
    K_l.append(k_l)
    n_l = ((2 * r_l + 2) + np.sqrt((-2 * r_l - 2) ** 2 - 4 * (1-r_l) * (k_l ** 2 * (-(r_l-1)) - r_l + 1))) / (2 * ( 1 - r_l ))
    N_l.append(n_l)
anser = np.vstack([N,K])
anser_l = np.vstack([N_l,K_l])
np.savetxt('n_short', N)
np.savetxt('k_short', K)
np.savetxt('n_long', N_l)
np.savetxt('k_long', K_l)
np.save('ConstantNPY\ n_short', N)
np.save('ConstantNPY\ k_short',K)
np.save('ConstantNPY\ n_long', N_l)
np.save('ConstantNPY\ k_long',K_l)
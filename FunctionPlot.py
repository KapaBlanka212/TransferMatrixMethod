import MaterialConstant
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

const = MaterialConstant.MaterialConstant()
l = const.full_wavelength
np.savetxt('full_wavelength',l)
T_exp_full = const.T_exp_6
R_exp_full = const.R_exp_6
TR = np.load('TR6.npy')
T = TR[:,0]
R = TR[:,1]
G1 = plt.plot(l,T)
G2 = plt.plot(l,T_exp_full)
plt.xlabel(r"$\mcm$")
plt.ylabel(r"$T$")
G3 = plt.plot(l1,R)
G4 = plt.plot(l2,R_exp_full)
plt.xlabel(r"$\mcm$")
plt.ylabel(r"$R$")
plt.show(G1,G2)
plt.show(G3,G4)
import scipy.io as sio
import numpy as np
import MaterialConstant
c = MaterialConstant.MaterialConstant()
l = c.short_wavelenght
T1 = sio.loadmat('T1.mat')
T = np.array([T1])
R1 = sio.loadmat('R1.mat')
R = np.array([R1])
L1 = sio.loadmat('L1.mat')
L = np.array([L1])
indx = L.size
full1 = []
full = np.array([L,R,T])
print(full)
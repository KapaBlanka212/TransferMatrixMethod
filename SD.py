import numpy as np
import scipy.io as sio
FULL = sio.loadmat('ConstantMAT\Material6.mat')
full = np.array(FULL['Material6'])
T6_short = full[:,1]
T6_short = T6_short[::1000]
np.save('ConstantNPY\T6_short',T6_short)
R6_short = full[:,2]
R6_short = R6_short[::1000]
np.save('ConstantNPY\R6_short',R6_short)
#L1 = full[:,0]
#L1 = L1[::1000]
#np.save('wl',L1)
#FULL = sio.loadmat('Constant\Material4.mat')
#full = np.array(FULL['Material4'])
#T6_short = full[:,1]
#T6_short = T6_short[::1000]
#np.save('T6_short',T6_short)
#R6_short = full[:,2]
#R6_short = R6_short[::1000]
#np.save('R6_short',R6_short)
#FULL = sio.loadmat('Constant\Material5.mat')
#full = np.array(FULL['Material5'])
#T5_short = full[:,1]
#T5_short = T5_short[::1000]
#np.save('T5_short',T5_short)
#R6_short = full[:,2]
#R5_short = R5_short[::1000]
#np.save('R5_short',R5_short)
#FULL = sio.loadmat('Constant\Material3.mat')
#full = np.array(FULL['Material3'])
#T3_short = full[:,1]
#T3_short = T3_short[::1000]
#np.save('T3_short',T3_short)
#R3_short = full[:,2]
#R3_short = R3_short[::1000]
#np.save('R3_short',R3_short)
#FULL = sio.loadmat('Constant\Material2.mat')
#full = np.array(FULL['Material2'])
#T2_short = full[:,1]
#T2_short = T2_short[::1000]
#np.save('T2_short',T2_short)
#R2_short = full[:,2]
#R2_short = R6_short[::1000]
#np.save('R2_short',R2_short)
#FULL = sio.loadmat('Constant\Material1.mat')
#full = np.array(FULL['Material1'])
#T1_short = full[:,1]
#T1_short = T1_short[::1000]
#np.save('T1_short',T1_short)
#R1_short = full[:,2]
#R1_short = R1_short[::1000]
#np.save('R1_short',R1_short)
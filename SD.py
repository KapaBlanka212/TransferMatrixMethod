import numpy as np
import scipy.io as sio
FULL13 = sio.loadmat('ConstantMAT\M6_long.mat')
full13 = np.array(FULL13['M6_long'])
T6_long = np.array(full13[:,1])
np.save('ConstantNPY\T6_long',T6_long)
R6_long = np.array(full13[:,2])
np.save('ConstantNPY\R6_long',R6_long)
FULL14 = sio.loadmat('ConstantMAT\M6_short.mat')
print(FULL14)
full14 = np.array(FULL14['M6_short'])
T6_short = np.array(full14[:,1])
print(T6_short,T6_short.shape)
T6_short = T6_short[::1000]
np.save('ConstantNPY\T6_short',T6_short)
R6_short = np.array(full14[:,2])
print(R6_short,R6_short.shape)
R6_short = R6_short[::1000]
print(R6_short)
np.save('ConstantNPY\R6_short',R6_short)

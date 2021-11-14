import numpy as np
import scipy.io as sio
FULL1 = sio.loadmat('ConstantMAT\M1_long.mat')
full1 = np.array(FULL1['M1_long'])
T1_long = np.array(full1[:,1])
np.save('ConstantNPY\T1_long',T1_long)
R1_long = np.array(full1[:,2])
np.save('ConstantNPY\R1_long',R1_long)
FULL2 = sio.loadmat('ConstantMAT\M1_short.mat')
full2 = np.array(FULL2['M1_short'])
T1_short = np.array(full2[:,1])
T1_short = T1_short[::1000]
np.save('ConstantNPY\T1_short',T1_short)
R1_short = np.array(full2[:,2])
R1_short = R1_short[::1000]
np.save('ConstantNPY\R1_short',R1_short)
FULL3 = sio.loadmat('ConstantMAT\M2_long.mat')
full3 = np.array(FULL3['M2_long'])
T2_long = np.array(full3[:,1])
np.save('ConstantNPY\T2_long',T2_long)
R2_long = np.array(full3[:,2])
np.save('ConstantNPY\R2_long',R2_long)
FULL4 = sio.loadmat('ConstantMAT\M2_short.mat')
full4 = np.array(FULL4['M2_short'])
T2_short = np.array(full4[:,1])
T2_short = T2_short[::1000]
np.save('ConstantNPY\T2_short',T2_short)
R2_short = np.array(full4[:,2])
R2_short = R2_short[::1000]
np.save('ConstantNPY\R2_short',R2_short)
FULL5 = sio.loadmat('ConstantMAT\M3_long.mat')
full5 = np.array(FULL5['M3_long'])
T3_long = np.array(full5[:,1])
np.save('ConstantNPY\T3_long',T3_long)
R3_long = np.array(full5[:,2])
np.save('ConstantNPY\R3_long',R3_long)
FULL6 = sio.loadmat('ConstantMAT\M3_short.mat')
full6 = np.array(FULL6['M3_short'])
T3_short = np.array(full6[:,1])
T3_short = T3_short[::1000]
np.save('ConstantNPY\T3_short',T3_short)
R3_short = np.array(full6[:,2])
R3_short = R3_short[::1000]
np.save('ConstantNPY\R3_short',R3_short)
FULL7 = sio.loadmat('ConstantMAT\M4_long.mat')
full7 = np.array(FULL7['M4_long'])
T4_long = np.array(full7[:,1])
np.save('ConstantNPY\T4_long',T4_long)
R4_long = np.array(full7[:,2])
np.save('ConstantNPY\R4_long',R4_long)
FULL8 = sio.loadmat('ConstantMAT\M4_short.mat')
full8 = np.array(FULL8['M4_short'])
T4_short = np.array(full8[:,1])
T4_short = T4_short[::1000]
np.save('ConstantNPY\T4_short',T4_short)
R4_short = np.array(full8[:,2])
R4_short = R4_short[::1000]
np.save('ConstantNPY\R4_short',R4_short)
FULL9 = sio.loadmat('ConstantMAT\M5_long.mat')
full9 = np.array(FULL9['M5_long'])
T5_long = np.array(full9[:,1])
np.save('ConstantNPY\T5_long',T5_long)
R5_long = np.array(full9[:,2])
np.save('ConstantNPY\R5_long',R5_long)
FULL10 = sio.loadmat('ConstantMAT\M5_short.mat')
full10 = np.array(FULL10['M5_short'])
T5_short = np.array(full10[:,1])
T5_short = T5_short[::1000]
np.save('ConstantNPY\T5_short',T5_short)
R5_short = np.array(full10[:,2])
R5_short = R5_short[::1000]
np.save('ConstantNPY\R5_short',R5_short)
#import numpy as np
#import scipy.io as sio
import scipy.integrate as integrate
#FULL13 = sio.loadmat('ConstantMAT\M6_long.mat')
#full13 = np.array(FULL13['M6_long'])
#T6_long = np.array(full13[:,1])
#np.save('ConstantNPY\T6_long',T6_long)
#R6_long = np.array(full13[:,2])
#np.save('ConstantNPY\R6_long',R6_long)
##FULL14 = sio.loadmat('ConstantMAT\M6_short.mat')
##print(FULL14)
#full14 = np.array(FULL14['M6_short'])
#L = np.array(full14[:,0])
#print(T6_short,T6_short.shape)
#L = L[::100]
#np.savetxt('wl',L)
#np.save('ConstantNPY\wl',L)
#T6_short = np.array(full14[:,1])
#print(T6_short,T6_short.shape)
#T6_short = T6_short[::100]
#np.save('ConstantNPY\T6_short',T6_short)
#R6_short = np.array(full14[:,2])
#print(R6_short,R6_short.shape)
#R6_short = R6_short[::100]
#print(R6_short.shape)
#np.save('ConstantNPY\R6_short',R6_short)
#print(A)
#print(A1)
#print(A-A1)
def x(x):
    x_out = x**2
    return x_out
out = integrate.quad(x,0,1)
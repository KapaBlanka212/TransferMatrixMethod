# this code need for reading and rewriting some .mat file
from scipy.io import loadmat
import numpy as np
m4 = loadmat('ConstantMAT\M4_short.mat')
print(m4)
wave_transmittance_refraction = np.array(m4['M4_short'])
print(wave_transmittance_refraction, wave_transmittance_refraction.shape)
wavelength = wave_transmittance_refraction[:, 0]
wavelength = wavelength[::100]
print(wavelength, wavelength.shape)
transmittance = wave_transmittance_refraction[:, 1]
transmittance = transmittance[::100]
refraction = wave_transmittance_refraction[:, 2]
refraction = refraction[::100]
print(refraction, refraction.size)
np.save('ConstantNPY\R4_short.npy',refraction)

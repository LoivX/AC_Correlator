import sys
sys.path.append(r"C:\Users\gioch\OneDrive\Desktop\Università\terzo anno\Tesi")
from astrocook.astrocook.functions import lines_voigt
import astropy as ap
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.interpolate import interp1d


spectrum = Table.read('J0942UVES_original_CIV_from_list_spec.dat', format='ascii')
spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

#Parameters definition
wav_start = 154.8
wav_end = 155.1

z = 0
logN = 12
b = 5
btur = 0
ion = 'CIV'

z_start = spectrum['wavelength'][0]/154.8 - 1
z_end = spectrum['wavelength'][-1]/155.1 - 1
dz = 1e-3

#Model definition
x = np.linspace(wav_start, wav_end, 1000)
y = lines_voigt(x, z, logN, b, btur, ion)

cor_all = np.array([])


#correlation calculation
for z in np.arange(z_start, z_end, dz):
    
    #moving the model to the redshift z
    model = Table([x*(1 + z), y], names=['wavelength', 'flux'])
    
    #identifying the range of the spectrum that is covered by the model
    model_wavelength_min = model['wavelength'].min()
    model_wavelength_max = model['wavelength'].max()
    
    mask = (spectrum['wavelength'] > model_wavelength_min) & (spectrum['wavelength'] < model_wavelength_max)    
    
    #selecting the data interval covered by the model
    spec_chunk = Table([spectrum['wavelength'][mask], spectrum['flux'][mask]], names=['wavelength', 'flux'])

    #interpolating the model to the data wavelength
    interpolate = interp1d(model['wavelength'], model['flux'], kind='linear')
    interpolated_flux = interpolate(spec_chunk['wavelength'])

    cor = np.correlate(interpolated_flux, spec_chunk['flux'], mode='valid')
    print([z, cor])
    cor_all = np.append(cor_all, cor)


z_interval = np.arange(z_start, z_end, dz)

plt.plot(z_interval, cor_all)
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.show()
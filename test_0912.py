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

x = np.linspace(wav_start, wav_end, 247)

z = 0
logN = 12
b = 5
btur = 0
ion = 'CIV'

x = x*(1 + 2.71)

y = lines_voigt(x, 2.71, logN, b, btur, ion)


model = Table([x, y], names=['wavelength', 'flux'])
model['flux'][model['flux'] > 0.99] = np.nan

indecies = np.where(~np.isnan(model['flux']))

z_start = spectrum['wavelength'][0]/154.8 - 1
z_end = spectrum['wavelength'][-1]/155.1 - 1
dz = 1e-3

cor = np.zeros(len(spectrum['wavelength'])-247)

for i in range(len(spectrum['wavelength'])-247):
    spec_chunk = spectrum['flux'][i:i+247]
    #print(len(spec_chunk), len(model['flux']))
    
    cor[i] = np.correlate(spec_chunk[indecies], model['flux'][indecies], mode='valid')


print(len(cor), len(spectrum['wavelength']))
cor_table = Table([spectrum['wavelength'][0:-247], cor], names=['wavelength', 'correlation'])

plt.figure(figsize=(10, 6))
plt.xlabel('Wavelength')
plt.ylabel('Correlation')
plt.title('Correlation vs Wavelength')
plt.plot(spectrum['wavelength'], spectrum['flux']*11)
plt.plot(cor_table['wavelength'], cor_table['correlation'])
plt.show()
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


spectrum = Table.read('J0942UVES_original_CIV_from_list_spec_synth_spec.dat', format='ascii')
deabs_spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

wav_start = 154.8
wav_end = 155.1

#model parametrs
z = 0
logN = 12
b = 5
btur = 0
ion = 'CIV'

# model definition
x = np.linspace(wav_start, wav_end, 1000)
y = lines_voigt(x, z, logN, b, btur, ion)
model = Table([x, y], names=['wavelength', 'flux'])

#other parameters
threshold = 0.999
z_start = deabs_spectrum['wavelength'][0]/154.8 - 1
z_end = deabs_spectrum['wavelength'][-1]/155.1 - 1
dz = 1e-5

#get indicies function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

cor_all = np.array([])

for z in np.arange(z_start, z_end, dz):
    
    #moving the model to the redshift z
    new_model = Table([model['wavelength']*(1 + z), model['flux']], names=['wavelength', 'flux'])
    
    #identifying the range of the spectrum that is covered by the model
    model_wavelength_min = new_model['wavelength'].min()
    model_wavelength_max = new_model['wavelength'].max()
    
    mask = (deabs_spectrum['wavelength'] > model_wavelength_min) & (deabs_spectrum['wavelength'] < model_wavelength_max)    
    
    #selecting the data interval covered by the model
    spec_chunk = Table([deabs_spectrum['wavelength'][mask], deabs_spectrum['flux'][mask]], names=['wavelength', 'flux'])

    #interpolating the model to the data wavelength
    interpolate = interp1d(new_model['wavelength'], new_model['flux'], kind='linear')
    interpolated_flux = interpolate(spec_chunk['wavelength'])

    #identifying the indicies of the model that are below the threshold
    indicies, count = get_indicies(interpolated_flux, threshold)

    cor = np.correlate(interpolated_flux[indicies], spec_chunk['flux'][indicies], mode='valid')/count
    cor_all = np.append(cor_all, cor)


z_interval = np.arange(z_start, z_end, dz)

plt.plot(z_interval, cor_all)
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.show()
import sys
sys.path.append(r"d:\Università\terzo anno\Tesi")
from astrocook.astrocook.functions import lines_voigt
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm


spectrum = Table.read('testCIV_2_spec.dat', format='ascii')
spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

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
z_start = spectrum['wavelength'][0]/154.8 - 1
z_end = spectrum['wavelength'][-1]/155.1 - 1
dz = 1e-5

#get indicies function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

cor_all = np.array([])

for z in tqdm(np.arange(z_start, z_end, dz), desc='Calculating correlation: '):
    
    #moving the model to the redshift z
    new_model = Table([model['wavelength']*(1 + z), model['flux']], names=['wavelength', 'flux'])
    
    #identifying the range of the spectrum that is covered by the model
    model_wavelength_min = new_model['wavelength'].min()
    model_wavelength_max = new_model['wavelength'].max()
    
    mask = (spectrum['wavelength'] > model_wavelength_min) & (spectrum['wavelength'] < model_wavelength_max)    
    
    #selecting the data interval covered by the model
    spec_chunk = Table([spectrum['wavelength'][mask], spectrum['flux'][mask]], names=['wavelength', 'flux'])

    #interpolating the model to the data wavelength
    interpolate = interp1d(new_model['wavelength'], new_model['flux'], kind='linear')
    interpolated_flux = interpolate(spec_chunk['wavelength'])

    #identifying the indicies of the model that are below the threshold
    indicies, count = get_indicies(interpolated_flux, threshold)

    cor = np.correlate(interpolated_flux[indicies], spec_chunk['flux'][indicies], mode='valid')/count
    cor_all = np.append(cor_all, cor)


z_interval = np.arange(z_start, z_end, dz)

# Normalizing the correlation
cor_graph = (cor_all.max() - cor_all)/(cor_all.max() - cor_all.min())

# Function to convert bin to redshift
def bin_to_z(bin):
    return z_start + bin*dz

# Finding the peaks
peaks, properties = sps.find_peaks(cor_graph, height=0.1, prominence=0.1, width=0.01)
peaks = bin_to_z(peaks)   

syst_dict = {'CIV' : (155.0-154.8)/154.8} #definisco il dizionario degli elementi (modificabile per adattarla ad astrocook)


def is_ion(peaks_triplet, ion='CIV', tollerance= 2e-3):
        delta_z = [syst_dict[ion] * (1 + peaks_triplet[1]), syst_dict[ion] * (1 + peaks_triplet[2])]
        model_data_difference = np.abs(np.diff(peaks_triplet) - delta_z)
        if(np.all(model_data_difference <= tollerance)):
            return True
        else:
            return False

for i in range(len(peaks)):
        if(i == 0 or i == len(peaks) - 1):
            continue
        else:
            peaks_triplet = [peaks[i-1], peaks[i], peaks[i+1]]
            if(is_ion(peaks_triplet)):
                print("Possible system:", peaks_triplet[1])




plt.plot(z_interval, cor_graph)
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.show()
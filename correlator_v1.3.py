import sys
sys.path.append(r"d:\Università\terzo anno\Tesi")
from astrocook.astrocook.functions import lines_voigt
from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm


##FUNCTIONS##

# Function to convert bin to redshift
def bin_to_z(bin):
    return z_start + bin*dz

# Function to find the peaks studying the delta_z between the peaks
def find_peaks(peaks_table, ion = 'CIV', tolerance = 1.5e-3):
    possible_systems = []
    
    for peak in peaks_table:
        peak_triplet = peaks_table[peaks_table == peak]
        # Defining the theoretical redshifts of the sx and dx peaks
        delta_z = syst_dict[ion] * (1 + peak['z'])
        sx_z = peak['z'] - delta_z
        dx_z = peak['z'] + delta_z

        # Defining the masks to find the sx and dx peaks within the tolerance
        sx_mask = (peaks_table['z'] < sx_z + tolerance) & (peaks_table['z'] > sx_z - tolerance)
        dx_mask = (peaks_table['z'] < dx_z + tolerance) & (peaks_table['z'] > dx_z - tolerance)

        #identifying the sx and dx peaks
        sx_peak = peaks_table[sx_mask]
        dx_peak = peaks_table[dx_mask]
        
        # FILTERING
        if(peak['z'] in sx_peak['z']):
            sx_peak = sx_peak[sx_peak['z'] != peak['z']]
        if(peak['z'] in dx_peak['z']):
            dx_peak = dx_peak[dx_peak['z'] != peak['z']]
        
        mask = ((len(sx_peak) > 1) | (len(dx_peak) > 1)) & ((len(sx_peak) != 0) & (len(dx_peak) != 0))
        
        if((len(sx_peak) == 1) & (len(dx_peak) == 1)):
            peak_triplet = vstack([sx_peak, peak_triplet, dx_peak])
            possible_systems.append(round(peak_triplet['z'][1], 4))

        elif(mask):
            sx_peak = sx_peak[np.abs(delta_z - (peak['z'] - sx_peak['z'])).argmin()]
            dx_peak = dx_peak[np.abs(delta_z - (peak['z'] - dx_peak['z'])).argmin()]

            peak_triplet = vstack([sx_peak, peak_triplet, dx_peak])
            possible_systems.append(round(peak_triplet['z'][1], 4))

    #height filternig

    return possible_systems




##MAIN##

spectrum = Table.read('testCIV_3_spec.dat', format='ascii')
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


# Finding the peaks
peaks, properties = sps.find_peaks(cor_graph, height=0.05, prominence=0.05, width=0.01)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

syst_dict = {'CIV' : (155.0-154.8)/154.8} #definisco il dizionario degli elementi (modificabile per adattarla ad astrocook)

possible_systems = find_peaks(peaks_table)
print(possible_systems)

plt.plot(z_interval, cor_graph)
for system in possible_systems:
    plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.show()
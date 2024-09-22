import sys
sys.path.append(r"d:\Università\terzo anno\Tesi\astrocook")
from astrocook.functions import lines_voigt
from astrocook import vars
from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm


##FUNCTIONS##

#get indicies function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

# Function to convert bin to redshift
def bin_to_z(bin):
    return z_start + bin*dz

# Function to get the Δλ of the ion
def get_dlambda(ion):
    ion_components = {key: value for key, value in vars.xem_d.items() if ion in key}
    c1 = float(ion_components[list(ion_components.items())[0][0]].value)
    c2 = float(ion_components[list(ion_components.items())[1][0]].value)

    dlambda = (np.abs(c1 - c2))/c1

    return dlambda

# Function to find the peaks studying the delta_z between the peaks
def find_systems(peaks_table, ion = 'CIV', z_tolerance = 0.08e-3):
    possible_systems = []

    for peak in peaks_table[peaks_table['height'] > np.mean(cor_final) + np.std(cor_final)*3]:
        # Defining the theoretical redshifts of the sx and dx peaks
        delta_lambda = get_dlambda(ion)
        delta_z = delta_lambda * (1 + peak['z'])
        sx_z = peak['z'] - delta_z
        dx_z = peak['z'] + delta_z

        # Defining the masks to find the sx and dx peaks within the z_tolerance
        sx_mask = (peaks_table['z'] < sx_z + z_tolerance) & (peaks_table['z'] > sx_z - z_tolerance)
        dx_mask = (peaks_table['z'] < dx_z + z_tolerance) & (peaks_table['z'] > dx_z - z_tolerance) 
        height_mask = (peaks_table['height'] < peak['height'])
        not_peak_z_mask = (peaks_table['z'] != peak['z'])

        #identifying the sx and dx peaks
        sx_peak = peaks_table[sx_mask & height_mask & not_peak_z_mask]
        dx_peak = peaks_table[dx_mask & height_mask & not_peak_z_mask]

        # Saving the possible systems
        peak_found = False
        if((len(sx_peak) >= 1) & (len(dx_peak) >= 1)):
            peak_found = True

        if(peak_found & (not (peak['z'] in possible_systems))):
            print('\n Found possibile system at:', peak['z'])	
            possible_systems.append(round(peak['z'], 4))

    return possible_systems




##MAIN##

spectrum = Table.read('testCIV_4_spec.dat', format='ascii')
spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

wav_start = [154.8, 154.8, 155.06]
wav_end = [155.1, 154.85, 155.1]

#model parametrs
z = 0
logN = 12
b = 5
btur = 0
ion = 'CIV'


# models definition
models = [Table(), Table(), Table()]
for i in range(len(wav_start)):
    x = np.linspace(wav_start[i], wav_end[i], 1000)
    y = lines_voigt(x, z, logN, b, btur, ion)
    models[i] = Table([x, y], names=['wavelength', 'flux'])


#other parameters
threshold = 0.999
z_start = spectrum['wavelength'][0]/154.8 - 1
z_end = spectrum['wavelength'][-1]/155.1 - 1
#z_start = 2.56
#z_end = 2.63
dz = 1e-5


cor_all = [np.array([]), np.array([]), np.array([])]

print("\n 3 models correlation calculation: \n - model 0 = full doublet \n - model 1 = left peak \n - model 2 = right peak \n")

for i,model in enumerate(models):
    for z in tqdm(np.arange(z_start, z_end, dz), "Calculating Correlation with model {}".format(i)):
        #moving the model to the redshift z
        new_model = Table([model['wavelength']*(1 + z), model['flux']], names=['wavelength', 'flux'])

        mask = (spectrum['wavelength'] > new_model['wavelength'].min()) & (spectrum['wavelength'] < new_model['wavelength'].max())    
        
        #selecting the data interval covered by the model
        spec_chunk = Table([spectrum['wavelength'][mask], spectrum['flux'][mask]], names=['wavelength', 'flux'])

        #interpolating the model to the data wavelength
        interpolate = interp1d(new_model['wavelength'], new_model['flux'], kind='linear')
        interpolated_flux = interpolate(spec_chunk['wavelength'])

        #identifying the indicies of the model that are below the threshold
        indicies, count = get_indicies(interpolated_flux, threshold)

        cor = np.correlate(interpolated_flux[indicies], spec_chunk['flux'][indicies], mode='valid')/count
        cor_all[i] = np.append(cor_all[i], cor)

cor_final = (1-cor_all[0])*(1-cor_all[1])*(1-cor_all[2])
z_interval = np.arange(z_start, z_end, dz)


# Finding the peaks
peaks, properties = sps.find_peaks(cor_final, height=np.mean(cor_final) + np.std(cor_final)*2, prominence=0, width=0.01)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

possible_systems = find_systems(peaks_table)
print("Fount possible systems:", possible_systems)

plt.figure(figsize = (20, 12))
plt.plot(z_interval, cor_final)
plt.axhline(np.mean(cor_final), color='r', linestyle='--', alpha=0.5)
plt.axhline(np.std(cor_final)*3 + np.mean(cor_final), color='yellow', linestyle='--', alpha=0.5)
plt.axhline(np.std(cor_final)*2 + np.mean(cor_final), color='#CC7722', linestyle='--', alpha=0.5)

for system in possible_systems:
    plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'Probable Systems'])
plt.show()
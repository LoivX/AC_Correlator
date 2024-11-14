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
def find_systems(peaks_table, ion = 'CIV', z_tolerance = 0.1e-3, height_threshold = 3):
    
    for peak in peaks_table[peaks_table['height'] > np.mean(cor_final) + np.std(cor_final)*height_threshold]:
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

        # Removing the secondary peaks
        if((len(sx_peak) == 1) & (len(dx_peak) == 1)):
            for sx in sx_peak:
                peaks_table = peaks_table[peaks_table['z'] != sx['z']]
            for dx in dx_peak:
                peaks_table = peaks_table[peaks_table['z'] != dx['z']]
                
    return peaks_table['z']





##MAIN##

spectrum = Table.read('testCIV_8_spec.dat', format='ascii')
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

perc = 75

p0 = np.percentile(cor_all[0], perc)
p1 = np.percentile(cor_all[1], perc)
p2 = np.percentile(cor_all[2], perc)

cor_final = (p0 -cor_all[0])*(p1-cor_all[1])*(p2-cor_all[2])
z_interval = np.arange(z_start, z_end, dz)


# Finding the peaks
peaks, properties = sps.find_peaks(cor_final, height=np.mean(cor_final) + np.std(cor_final)*1.5, prominence=0, width=0.01, distance = 5e-4/dz)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

# defining the parameters for the system identification
height_threshold = 2.5
z_tolerance = 0.2e-3
ion = 'CIV'

possible_systems = []
f1tot = []

for ht in np.arange(2, 4, 0.05):
    for zt in np.arange(10e-5, 10e-4, 0.5e-5):
        possible_systems = find_systems(peaks_table, ion, zt, ht)

        synthetic_systems = [2.55646069, 2.58495365, 2.59680712, 2.60450324, 2.60796068, 2.61049208, 2.64539575, 2.64623804, 2.65191481, 2.66880955, 2.70345038, 2.71184958, 2.71576258, 2.7583452, 2.78713784, 2.81702964, 2.81859274, 2.88002692, 2.89315974, 2.93926941]

        tp = sum(1 for system in possible_systems if any(abs(system - syn) < 1.665*b/3e5 for syn in synthetic_systems))

        fp = len(possible_systems) - tp
        p = len(synthetic_systems)

        recall = tp / p * 100
        precision = tp / (tp+fp) * 100
        f1 = 2 * tp / (tp + fp + p) * 100

        #print('Recall:', recall, '%')
        #print('Precision:', precision, '%')
        print('\n parameters:', ht, zt)
        print('F1 score:', f1, '%')

        f1tot.append(f1)

print('Max F1:', max(f1tot))
print('Parameters:', np.arange(2, 4, 0.05)[f1tot.index(max(f1tot))], np.arange(10e-5, 10e-4, 0.5e-5)[f1tot.index(max(f1tot))])
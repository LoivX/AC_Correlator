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

#get indicies function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

# Function to convert bin to redshift
def bin_to_z(bin):
    return z_start + bin*dz

# Function to find the peaks studying the delta_z between the peaks
def find_systems(peaks_table, ion = 'CIV', z_tolerance = 0.05e-3, height_tolerance = 0.02):
    possible_systems = []
    peak_ratio = []
    for peak in peaks_table:
        peak_triplet = peaks_table[peaks_table == peak]
        # Defining the theoretical redshifts of the sx and dx peaks
        delta_z = syst_dict[ion][0] * (1 + peak['z'])
        sx_z = peak['z'] - delta_z
        dx_z = peak['z'] + delta_z

        print('\n \n picchi a sx e dx teorici:', sx_z, dx_z)

        # Defining the masks to find the sx and dx peaks within the z_tolerance
        sx_mask = (peaks_table['z'] < sx_z + z_tolerance) & (peaks_table['z'] > sx_z - z_tolerance)
        dx_mask = (peaks_table['z'] < dx_z + z_tolerance) & (peaks_table['z'] > dx_z - z_tolerance)

        #identifying the sx and dx peaks
        sx_peak = peaks_table[sx_mask]
        dx_peak = peaks_table[dx_mask]
        
        print('\n picchi a sx e dx trovati:', sx_peak['z'], dx_peak['z'])

        # FILTERING
        # Removing the current peak from the sx and dx peaks
        if(peak['z'] in sx_peak['z']):
            sx_peak = sx_peak[sx_peak['z'] != peak['z']]
        if(peak['z'] in dx_peak['z']):
            dx_peak = dx_peak[dx_peak['z'] != peak['z']]
        
        mask = ((len(sx_peak) > 1) | (len(dx_peak) > 1)) & ((len(sx_peak) != 0) & (len(dx_peak) != 0))
        
        # If the sx and dx peaks are unique, the triplet is complete
        if((len(sx_peak) == 1) & (len(dx_peak) == 1)):
            peak_triplet = vstack([sx_peak, peak_triplet, dx_peak])
        
        # If the sx and dx peaks are not unique, the peaks which distance from the current peak is the smallest are selected
        elif(mask):
            sx_peak = sx_peak[np.abs(sx_z - sx_peak['z']).argmin()]
            dx_peak = dx_peak[np.abs(dx_z - dx_peak['z']).argmin()]

            peak_triplet = vstack([sx_peak, peak_triplet, dx_peak])
        
        # height ratio check
        if(len(peak_triplet) == 3):
            peak_ratio = np.array([peak_triplet['height'][0]/peak_triplet['height'][1], peak_triplet['height'][2]/peak_triplet['height'][1]])
            print('\n peak ratio', peak_ratio)
            if(np.all(np.abs(peak_ratio - syst_dict[ion][1]) <= height_tolerance)):
                print('\n sistema trovato:', peak_triplet['z'])	
                possible_systems.append(round(peak_triplet['z'][1], 4))

    return possible_systems



##MAIN##

spectrum = Table.read('testCIV_5_spec.dat', format='ascii')
spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

wav_start = [154.8, 154.8, 155.06]
wav_end = [155.1, 154.85, 155.1]

#model parametrs
z = 0
logN = 11.5
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

cor_ratio = cor_all[0]/(cor_all[1]*cor_all[2])
z_interval = np.arange(z_start, z_end, dz)


# Finding the peaks
peaks, properties = sps.find_peaks(cor_ratio, height=1.033, width=0.01)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

syst_dict = {'CIV' : [(155.0-154.8)/154.8, [1.00, 0.99]]}   #definisco il dizionario degli elementi (da modificare per adattarla ad astrocook)
                                                            #il secondo elemento della lista è il rapporto tra le altezze dei picchi sx e dx riacvato dalla media sperimentale ma da ricavare in modo teorico            

probable_systems = find_systems(peaks_table)

plt.plot(z_interval, cor_ratio)
for system in probable_systems:
    plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift')
plt.show()
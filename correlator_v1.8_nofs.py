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


##MAIN##

spectrum = Table.read('testCIV_7_spec.dat', format='ascii')
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


perc = 75

p0 = np.percentile(cor_all[0], perc)
p1 = np.percentile(cor_all[1], perc)
p2 = np.percentile(cor_all[2], perc)

cor_final = (p0 -cor_all[0])*(p1-cor_all[1])*(p2-cor_all[2])
z_interval = np.arange(z_start, z_end, dz)


# Finding the peaks
peaks, properties = sps.find_peaks(cor_final, height=np.mean(cor_final) + np.std(cor_final)*2.5, prominence=0, width=0.01, distance = 5e-4/dz)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

possible_systems = bin_to_z(peaks)
print(possible_systems, len(possible_systems))


synthetic_systems = [2.57352431, 2.66422603, 2.51803598, 2.22357274, 2.58163493, 2.21013991, 2.30970016, 2.82904971, 2.60683895, 2.95009428, 2.88590944, 2.27320993, 2.20884315, 3.08239514, 2.73699688, 2.30842111, 2.22430456, 2.68757115, 2.71786475, 2.98868828]


for i, p in enumerate(synthetic_systems):
    synthetic_systems[i] = round(p, 4)

n=0

false_positive = []
for system in possible_systems:
    system = round(system, 4)
    if(system in synthetic_systems):
        n+=1
        print(system, 'found')
    else:
        false_positive.append(system)

completness = n/len(synthetic_systems) * 100

print(completness, r'% of the synthetic systems have been found')
print(100 - completness, r'% of the synthetic systems have not been found')
print(len(false_positive)/len(possible_systems)*100, r'% of the found systems are false positives')



#PLOTS
plt.figure(figsize = (20, 12))
plt.plot(z_interval, cor_final)
plt.axhline(np.mean(cor_final), color='r', linestyle='--', alpha=0.5)
plt.axhline(np.std(cor_final)*3 + np.mean(cor_final), color='yellow', linestyle='--', alpha=0.5)
plt.axhline(np.std(cor_final)*2 + np.mean(cor_final), color='#CC7722', linestyle='--', alpha=0.5)

for system in possible_systems:
    plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

plt.xlabel('Redshift (z)')
plt.ylabel('Correlation')
plt.title('Correlation vs Redshift (nofs)')
plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems'])
plt.show()
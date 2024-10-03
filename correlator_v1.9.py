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
import tkinter as tk
from tkinter import simpledialog


##FUNCTIONS##

#get indicies function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

# Function to convert bin to redshift
def bin_to_z(bin):
    return z_start + bin*dz


root = tk.Tk()
root.withdraw()


##MAIN##

spectrum = Table.read('testCIV_7_spec.dat', format='ascii')
spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

wav_start = [154.8, 154.8, 155.06]
wav_end = [155.1, 154.85, 155.1]


# model params
z = 0

input_dialog = tk.Toplevel(root)
input_dialog.title("Model Parameters")

tk.Label(input_dialog, text="logN:").grid(row=0, column=0)
logN_entry = tk.Entry(input_dialog)
logN_entry.grid(row=0, column=1)
logN_entry.insert(0, "12.0")

tk.Label(input_dialog, text="b:").grid(row=1, column=0)
b_entry = tk.Entry(input_dialog)
b_entry.grid(row=1, column=1)
b_entry.insert(0, "5.0")

tk.Label(input_dialog, text="btur:").grid(row=2, column=0)
btur_entry = tk.Entry(input_dialog)
btur_entry.grid(row=2, column=1)
btur_entry.insert(0, "0.0")

tk.Label(input_dialog, text="ion (e.g., CIV):").grid(row=3, column=0)
ion_entry = tk.Entry(input_dialog)
ion_entry.grid(row=3, column=1)
ion_entry.insert(0, "CIV")

def model_params():
    global logN, b, btur, ion
    logN = float(logN_entry.get())
    b = float(b_entry.get())
    btur = float(btur_entry.get())
    ion = ion_entry.get()
    input_dialog.destroy()

submit_button = tk.Button(input_dialog, text="Submit", command=model_params)
submit_button.grid(row=4, columnspan=2)

input_dialog.wait_window()


# models definition
models = [Table(), Table(), Table()]
for i in range(len(wav_start)):
    x = np.linspace(wav_start[i], wav_end[i], 1000)
    y = lines_voigt(x, z, logN, b, btur, ion)
    models[i] = Table([x, y], names=['wavelength', 'flux'])


#other parameters
input_dialog = tk.Toplevel(root)
input_dialog.title("Spectrum Parameters")

tk.Label(input_dialog, text="threshold:").grid(row=0, column=0)
logN_entry = tk.Entry(input_dialog)
logN_entry.grid(row=0, column=1)
logN_entry.insert(0, "0.999")

tk.Label(input_dialog, text="z_start:").grid(row=1, column=0)   
z_start_entry = tk.Entry(input_dialog)
z_start_entry.grid(row=1, column=1)
z_start_entry.insert(0, str(spectrum['wavelength'][0]/154.8 - 1))

tk.Label(input_dialog, text="z_end:").grid(row=2, column=0)
z_end_entry = tk.Entry(input_dialog)
z_end_entry.grid(row=2, column=1)
z_end_entry.insert(0, str(spectrum['wavelength'][-1]/155.1 - 1))

tk.Label(input_dialog, text="dz:").grid(row=3, column=0)
dz_entry = tk.Entry(input_dialog)
dz_entry.grid(row=3, column=1)
dz_entry.insert(0, "1e-5")

def spec_params():
    global threshold, z_start, z_end, dz
    threshold = float(logN_entry.get())
    z_start = float(z_start_entry.get())
    z_end = float(z_end_entry.get())
    dz = float(dz_entry.get())
    input_dialog.destroy()

submit_button = tk.Button(input_dialog, text="Submit", command=spec_params)
submit_button.grid(row=4, columnspan=2)

input_dialog.wait_window()


#finding peak params
input_dialog = tk.Toplevel(root)
input_dialog.title("Finding Peaks Parameters")

tk.Label(input_dialog, text="Height tolerance:").grid(row=0, column=0)
logN_entry = tk.Entry(input_dialog)
logN_entry.grid(row=0, column=1)
logN_entry.insert(0, "2.5")

tk.Label(input_dialog, text="Prominence:").grid(row=1, column=0)
z_start_entry = tk.Entry(input_dialog)
z_start_entry.grid(row=1, column=1)
z_start_entry.insert(0, "0")

tk.Label(input_dialog, text="Width:").grid(row=2, column=0)
z_end_entry = tk.Entry(input_dialog)
z_end_entry.grid(row=2, column=1)
z_end_entry.insert(0, "0.01")

tk.Label(input_dialog, text="Distance:").grid(row=3, column=0)
dz_entry = tk.Entry(input_dialog)
dz_entry.grid(row=3, column=1)
dz_entry.insert(0, "5e-4")

def peak_params():
    global height_tol, prominence, width, distance
    height_tol = float(logN_entry.get())
    prominence = float(z_start_entry.get())
    width = float(z_end_entry.get())
    distance = float(dz_entry.get())/dz
    input_dialog.destroy()

submit_button = tk.Button(input_dialog, text="Submit", command=peak_params)
submit_button.grid(row=4, columnspan=2)

input_dialog.wait_window()


#correlation calculation

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

print(distance, type(distance))
# Finding the peaks
peaks, properties = sps.find_peaks(cor_final, np.mean(cor_final) + np.std(cor_final)*height_tol, 0, distance, prominence, width)
peaks_table = Table([bin_to_z(peaks), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips']), bin_to_z(properties['right_ips']), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

possible_systems = bin_to_z(peaks)
print(possible_systems, len(possible_systems))


#completness calculation
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
plt.title('Correlation vs Redshift')
plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems'])
plt.show()
import sys
sys.path.append(r"C:\Users\simon\Documents\Tesi\astrocook")
from astrocook.functions import lines_voigt, convolve_simple
from astrocook import vars
from astropy.table import Table
import numpy as np
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt

## FUNCTION ##
def plot_correlation(cor_final, z_interval, possible_systems, spectrum, threshold):
    #FIRST PLOT: spectrum with possible systems
    plt.figure(figsize=(20, 12))
    plt.plot(spectrum['x'], spectrum['y'])
    for system in possible_systems:
        plt.axvline(x=154.8204 * (1+system), color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Spectrum with possible Systems')

    #SECOND PLOT: correlation vs redshift
    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)
    plt.axhline(threshold, color='yellow', linestyle='--', alpha=0.5)
    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation', rf'Threshold', 'Possible Systems'])
    plt.show()

def psf_gauss(x, resol):
        if len(x)==0:
            return []
        c = x[len(x)//2]
        sigma = c / resol * 4.246609001e-1

        psf = np.exp(-0.5*((x-c) / sigma)**2)
        psf = psf[np.where(psf > 1e-6)]
        return psf

# Main function
def correlator(spectrum, plain_spec, resol, logN, b, ion, dz, n):
    # Define the region of interest
    ion_components = {key: value for key, value in vars.xem_d.items() if ion in key}
    c1, c2 = (float(ion_components[key].value) for key in list(ion_components)[:2])

    # defining complete model
    x = np.linspace(c1 - 4*1.665*b*c1/3e5, c2 + 4*1.665*b*c2/3e5, 1000)
    y = convolve_simple(lines_voigt(x, 0, logN, b, 0, ion), psf_gauss(x, resol))
    models = [Table([x, y], names=['x', 'y'])]

    # defining partial models
    for c in [c1, c2]:
        x = np.linspace(c - 4*1.665*b*c/3e5, c + 4*1.665*b*c/3e5, 1000)
        y = convolve_simple(lines_voigt(x, 0, logN, b, 0, ion), psf_gauss(x, resol))
        models.append(Table([x, y], names=['x', 'y']))

    z_start = spectrum['x'][0] / c1 - 1
    z_end = spectrum['x'][-1] / c2 - 1

    specs = [plain_spec, spectrum]

    for s, spec in enumerate(specs):
        cor_all = [np.array([]),np.array([]),np.array([])]
        for i,model in enumerate(models):
            for z in tqdm(np.arange(z_start, z_end, dz), f"Calculating Correlation with model {i}"):
                # Defining the interval on the spectrum in which the model is present
                mask = (spec['x'] > (model['x'].min())*(1+z)) & (spec['x'] <  (model['x'].max())*(1+z))
                spec_chunk = spec[mask]

                # Interpolating the model to the data x
                interpolate = interp1d(model['x'], model['y'], kind='linear')
                interpolated_flux = interpolate(spec_chunk['x'] / (1+z))

                # Calculating correlation
                cor = np.correlate(1-interpolated_flux, 1-spec_chunk['y'], mode='valid') / len(interpolated_flux)
                cor_all[i] = np.append(cor_all[i], cor)

        # Removing secondary peaks
        cor_final = (cor_all[0]) * (cor_all[1]) * (cor_all[2])

        if(s == 0):
            threshold = np.std(cor_final)*n + np.mean(cor_final)
        
        else:
            peaks , _ = sps.find_peaks(cor_final, height = threshold, prominence=0, width=0.01, distance=5e-4 / dz)
            if(len(peaks) == 0):
                print("Could not find any peaks :(")
            else:
                print(z_start + peaks*dz)
            
            plot_correlation(cor_final, np.arange(z_start, z_end, dz), z_start + peaks * dz, spectrum, threshold)
        
            return cor_final, np.arange(z_start, z_end, dz), z_start + peaks * dz
import sys
sys.path.append(r"d:\Università\terzo anno\Tesi\astrocook")
from astrocook.functions import lines_voigt, convolve_simple
from astrocook import vars
from astropy.table import Table
import numpy as np
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm

##FUNCTIONS##

# Get indices function
def get_indicies(spec_chunk, flux, threshold, ion, i, z):
    if i == 0:
        ion_components = {key: value for key, value in vars.xem_d.items() if ion in key}
        c1 = float(ion_components[list(ion_components.items())[0][0]].value)
        c2 = float(ion_components[list(ion_components.items())[1][0]].value)
        mask = ((spec_chunk['wavelength'] > c1*(1+z)) & (spec_chunk['wavelength'] < c2*(1+z))) | ((spec_chunk['wavelength'] < c1*(1+z)) & (spec_chunk['wavelength'] > c2*(1+z)) & (flux < 1- threshold))

        indicies = np.where(mask)[0]
    else:
        indicies = np.where((flux < 1- threshold))[0]
    
    count = len(indicies)
        
    return indicies, count

# Function to convert bin to redshift
def bin_to_z(bin, z_start, dz):
    return z_start + bin * dz

def psf_gauss(x, resol):
        if len(x)==0:
            return []
        c = x[len(x)//2]
        sigma = c / resol * 4.246609001e-1

        psf = np.exp(-0.5*((x-c) / sigma)**2)
        psf = psf[np.where(psf > 1e-6)]
        return psf

def correlator(spectrum_file, resol, wav_start, wav_end, logN, b, btur, ion, dz, perc):
    # Load spectrum
    spectrum = Table.read(spectrum_file, format='ascii')
    threshold = np.std(spectrum['dy'])

    spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])
        
    # Define models
    models = [Table(), Table(), Table()]
    for i in range(len(wav_start)):
        x = np.linspace(wav_start[i], wav_end[i], 1000)
        y = convolve_simple(lines_voigt(x, 0, logN, b, btur, ion), psf_gauss(x, resol))
        models[i] = Table([x, y], names=['wavelength', 'flux'])

    # Other parameters
    z_start = spectrum['wavelength'][0] / 154.8 - 1
    z_end = spectrum['wavelength'][-1] / 155.1 - 1

    cor_all = [np.array([]), np.array([]), np.array([])]

    for i, model in enumerate(models):
        for z in tqdm(np.arange(z_start, z_end, dz), f"Calculating Correlation with model {i}"): 
            # Defining the interval in which the model is present
            interval = [(model['wavelength'].min())*(1+z), (model['wavelength'].max())*(1+z)]    
            mask = (spectrum['wavelength'] > interval[0]) & (spectrum['wavelength'] < interval[1])

            # Selecting the data interval covered by the model
            spec_chunk = spectrum[mask]

            # Interpolating the model to the data wavelength
            interpolate = interp1d(model['wavelength']*(1 + z), model['flux'], kind='linear')
            interpolated_flux = interpolate(spec_chunk['wavelength'])

            # Identifying the indices of the model that are below the threshold
            indicies, count = get_indicies(spec_chunk, interpolated_flux, threshold, ion, i, z)

            cor = np.correlate(interpolated_flux[indicies], spec_chunk['flux'][indicies], mode='valid') / count
            cor_all[i] = np.append(cor_all[i], cor)

    # Percentile-based thresholding
    p0 = np.percentile(cor_all[0], perc)
    p1 = np.percentile(cor_all[1], perc)
    p2 = np.percentile(cor_all[2], perc)

    #cor_final = (p0 - cor_all[0]) * (p1 - cor_all[1]) * (p2 - cor_all[2])
    cor_final = p0 - cor_all[0]
    z_interval = np.arange(z_start, z_end, dz)

    # Finding the peaks
    peaks, properties = sps.find_peaks(cor_final, height=np.mean(cor_final) + np.std(cor_final) * 2.5, prominence=0, width=0.01, distance=5e-4 / dz)
    peaks_table = Table([bin_to_z(peaks, z_start, dz), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips'], z_start, dz), bin_to_z(properties['right_ips'], z_start, dz), properties['width_heights'], properties['prominences']],
                        names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])

    return cor_final, z_interval, peaks_table, models, spectrum

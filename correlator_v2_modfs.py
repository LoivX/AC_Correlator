import sys
sys.path.append(r"d:\Università\terzo anno\Tesi\astrocook")
from astrocook.functions import lines_voigt
from astrocook import vars
from astropy.table import Table
import numpy as np
import scipy.signal as sps
from scipy.interpolate import interp1d
from tqdm import tqdm

##FUNCTIONS##

# Get indices function
def get_indicies(flux, threshold):
    indicies = np.where(~(flux > threshold))[0]
    count = len(indicies)
    return indicies, count

# Function to convert bin to redshift
def bin_to_z(bin, z_start, dz):
    return z_start + bin * dz

# Function to get the Δλ of the ion
def get_dlambda(ion):
    ion_components = {key: value for key, value in vars.xem_d.items() if ion in key}
    c1 = float(ion_components[list(ion_components.items())[0][0]].value)
    c2 = float(ion_components[list(ion_components.items())[1][0]].value)

    dlambda = (np.abs(c1 - c2))/c1

    return dlambda

# Function to find the peaks studying the delta_z between the peaks
def find_systems(peaks_table, ion = 'CIV', z_tolerance = 0.1e-3, height_threshold = 3, cor_final = []):
    
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
                
    return peaks_table


def correlator(spectrum_file, wav_start, wav_end, logN, b, btur, ion, threshold, dz, perc, height_threshold, z_tolerance):
    # Load spectrum
    spectrum = Table.read(spectrum_file, format='ascii')
    spectrum = Table([spectrum['x'], spectrum['y']], names=['wavelength', 'flux'])

    # Define models
    models = [Table(), Table(), Table()]
    for i in range(len(wav_start)):
        x = np.linspace(wav_start[i], wav_end[i], 1000)
        y = lines_voigt(x, 0, logN, b, btur, ion)
        models[i] = Table([x, y], names=['wavelength', 'flux'])

    # Other parameters
    z_start = spectrum['wavelength'][0] / 154.8 - 1
    z_end = spectrum['wavelength'][-1] / 155.1 - 1

    cor_all = [np.array([]), np.array([]), np.array([])]

    for i, model in enumerate(models):
        for z in tqdm(np.arange(z_start, z_end, dz), f"Calculating Correlation with model {i}"):
            new_model = Table([model['wavelength'] * (1 + z), model['flux']], names=['wavelength', 'flux'])

            mask = (spectrum['wavelength'] > new_model['wavelength'].min()) & (spectrum['wavelength'] < new_model['wavelength'].max())

            # Selecting the data interval covered by the model
            spec_chunk = Table([spectrum['wavelength'][mask], spectrum['flux'][mask]], names=['wavelength', 'flux'])

            # Interpolating the model to the data wavelength
            interpolate = interp1d(new_model['wavelength'], new_model['flux'], kind='linear')
            interpolated_flux = interpolate(spec_chunk['wavelength'])

            # Identifying the indices of the model that are below the threshold
            indicies, count = get_indicies(interpolated_flux, threshold)

            cor = np.correlate(interpolated_flux[indicies], spec_chunk['flux'][indicies], mode='valid') / count
            cor_all[i] = np.append(cor_all[i], cor)

    # Percentile-based thresholding
    p0 = np.percentile(cor_all[0], perc)
    p1 = np.percentile(cor_all[1], perc)
    p2 = np.percentile(cor_all[2], perc)

    cor_final = (p0 - cor_all[0]) * (p1 - cor_all[1]) * (p2 - cor_all[2])
    z_interval = np.arange(z_start, z_end, dz)

    # Finding the peaks
    peaks, properties = sps.find_peaks(cor_final, height=np.mean(cor_final) + np.std(cor_final)*1.5, prominence=0, width=0.01, distance = 5e-4/dz)
    peaks_table = Table([bin_to_z(peaks, z_start, dz), properties['peak_heights'], properties['widths'], bin_to_z(properties['left_ips'], z_start, dz), bin_to_z(properties['right_ips'], z_start, dz), properties['width_heights'], properties['prominences']], names=['z', 'height', 'fwhm', 'left_z', 'right_z', 'half_max', 'prominence'])


    peaks_table = find_systems(peaks_table, ion, z_tolerance , height_threshold, cor_final)

    return cor_final, z_interval, peaks_table

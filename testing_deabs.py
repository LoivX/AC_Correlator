# %%
import matplotlib.pyplot as plt
from correlator_v3_deabs import deabs_correlator
import numpy as np
from astropy.table import Table


# %%
def plot_correlation(cor_final, z_interval, possible_systems, spectrum_file, threshold):
    spectrum = Table.read(spectrum_file, format='ascii')['x' , 'deabs']

    plt.figure(figsize=(20, 12))
    plt.plot(spectrum['x'], spectrum['deabs'])

    for system in possible_systems:
        plt.axvline(x=154.8204 * (1+system), color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Deep Spectrum UVES with possible Systems')

    
    
    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)

    plt.axhline(threshold, color='yellow', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation', r'3$\sigma$', 'Possible Systems'])
    plt.show()


# %%
if __name__ == '__main__':

    plain_spec = r'C:\Users\simon\Documents\Tesi\AC_Correlator\spectra\plain_spec_320_spec.dat'
    spectrum_file = r'C:\Users\simon\Documents\Tesi\AC_Correlator\spectra\J0942UVES_original_CIV_AND_METALS_spec.dat'
    
    ## PARAMETERS ##
    # Model parameters
    logN = 10.5              
    b = 5
    ion = 'CIV'

    # Other parameters
    dz= 1e-5
    resol = 45000

    thrs = deabs_correlator(plain_spec, resol, logN, b, ion, dz, 1)
    print(thrs)
    cor_final, z_interval, peaks = deabs_correlator(spectrum_file, resol, logN, b, ion, dz, thrs)

    print(peaks)

# %%
    plot_correlation(cor_final, z_interval, peaks, spectrum_file, thrs)



    
# %%
import matplotlib.pyplot as plt
from correlator_v2 import correlator
import numpy as np
from scipy.interpolate import interp1d


# %%
def plot_model_vs_data(model, spectrum, z):
    interval = [(model['wavelength'].min())*(1+z), (model['wavelength'].max())*(1+z)]    
    mask = (spectrum['wavelength'] > interval[0]) & (spectrum['wavelength'] < interval[1])

    # Selecting the data interval covered by the model
    spec_chunk = spectrum[mask]

    # Interpolating the model to the data wavelength
    interpolate = interp1d(model['wavelength']*(1 + z), model['flux'], kind='linear')
    interpolated_flux = interpolate(spec_chunk['wavelength'])

    
    plt.figure()
    plt.plot(spec_chunk['wavelength'], spec_chunk['flux']) 
    plt.plot(spec_chunk['wavelength'], interpolated_flux)
    plt.savefig(rf'D:\Università\terzo anno\Tesi\Immagini\model over data\model_vs_data_{z}.png')

#%%

def plot_correlation(cor_final, z_interval, possible_systems, synthetic_systems):
    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)
    for system in synthetic_systems:
        plt.axvline(x=system, color='r', alpha=0.3)

    plt.axhline(np.std(cor_final) * 3 + np.mean(cor_final), color='yellow', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems', 'synthetic Systems'])
    plt.show()

def RPF1(peaks_table, synthetic_systems, b):
    tp = sum(1 for system in peaks_table['z'] if any(abs(system - syn)/(1+syn) < 1.665*b/3e5 for syn in synthetic_systems))
    
    fp = len(peaks_table['z']) - tp
    p = len(synthetic_systems)

    recall = tp / p * 100
    precision = tp / (tp+fp) * 100
    f1 = 2 * tp / (tp + fp + p) * 100

    not_identified = [system for system in synthetic_systems if not any(abs(system - syn)/(1+syn) < 1.665*b/3e5 for syn in peaks_table['z'])]

    print('Recall:', recall, '%')
    print('Precision:', precision, '%')
    print('F1 score:', f1, '%')
    print('Not identified:', not_identified)

# %%
if __name__ == '__main__':

    spectrum_file = r'D:\Università\terzo anno\Tesi\AC_Correlator\test_mos2_spec.dat'
    
    ## PARAMETERS ##
    # Model parameters
    wav_start = [154.8, 154.8, 155.06]
    wav_end = [155.1, 154.85, 155.1]
    z = 0
    logN = 12.4
    b = 5
    btur = 0
    ion = 'CIV'

    # Other parameters
    dz= 1e-5
    perc= 75
    resol = 45000
    


    cor_final, z_interval, peaks_table, models, spectrum = correlator(spectrum_file, resol, wav_start, wav_end, logN, b, btur, ion, dz, perc)



    # Completness calculation
    synthetic_systems = [2.66462998, 2.55587099, 2.96921979, 2.76635491, 2.7774652,  2.70447149, 2.96940649, 2.73301883, 2.8231059,  2.70423362, 2.61945458, 2.9466871, 2.95230607, 2.71348916, 2.98043051, 2.63200771, 2.63711607, 2.69366724, 2.86461042, 2.88744127]
# %%
    RPF1(peaks_table, synthetic_systems, b)
# %%
   # for sys in synthetic_systems:
   #     plot_model_vs_data(models[0], spectrum, sys)
# %%
    plot_correlation(cor_final, z_interval, peaks_table['z'], synthetic_systems)



    
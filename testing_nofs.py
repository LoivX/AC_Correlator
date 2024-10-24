# %%
import matplotlib.pyplot as plt
from correlator_v3 import correlator
import numpy as np
from scipy.interpolate import interp1d


# %%
def plot_model_vs_data(model, spectrum, z):
    interval = [(model['x'].min())*(1+z), (model['x'].max())*(1+z)]    
    mask = (spectrum['x'] > interval[0]) & (spectrum['x'] < interval[1])

    # Selecting the data interval covered by the model
    spec_chunk = spectrum[mask]

    # Interpolating the model to the data x
    interpolate = interp1d(model['x']*(1 + z), model['y'], kind='linear')
    interpolated_flux = interpolate(spec_chunk['x'])

    
    plt.figure()
    plt.plot(spec_chunk['x'], spec_chunk['y']) 
    plt.plot(spec_chunk['x'], interpolated_flux)
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

def RPF1(peaks, synthetic_systems, b):
    tp = sum(1 for system in peaks if any(abs(system - syn)/(1+syn) < 1.665*b/3e5 for syn in synthetic_systems))
    
    fp = len(peaks) - tp
    p = len(synthetic_systems)

    recall = tp / p * 100
    precision = tp / (tp+fp) * 100
    f1 = 2 * tp / (tp + fp + p) * 100

    not_identified = [system for system in synthetic_systems if not any(abs(system - syn)/(1+syn) < 1.665*b/3e5 for syn in peaks)]

    print('Recall:', recall, '%')
    print('Precision:', precision, '%')
    print('F1 score:', f1, '%')
    print('Not identified:', not_identified)

# %%
if __name__ == '__main__':

    spectrum_file = r'D:\Università\terzo anno\Tesi\AC_Correlator\test_24_200_spec.dat'
    
    ## PARAMETERS ##
    # Model parameters
    logN = 12.5
    b = 5
    ion = 'CIV'

    # Other parameters
    dz= 0.8e-5
    resol = 45000

    cor_final, z_interval, peaks = correlator(spectrum_file, resol, logN, b, ion, dz)

    # Completness calculation
    synthetic_systems = [2.51097581, 2.53316622, 2.53765678, 2.55659701, 2.5738651, 2.58102128, 2.60045534, 2.64506485, 2.69327996, 2.72935909, 2.75362322, 2.76679574, 2.78899205, 2.813958, 2.84173642, 2.84820465, 2.95841746, 2.96369678, 2.97805738, 2.99442473]

# %%
    RPF1(peaks, synthetic_systems, b)
# %%
    #for sys in synthetic_systems:
    #    plot_model_vs_data(models[0], spectrum, sys)
# %%
    plot_correlation(cor_final, z_interval, peaks, synthetic_systems)



    
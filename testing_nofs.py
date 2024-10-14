import matplotlib.pyplot as plt
from correlator_v2 import correlator
import numpy as np

def plot_correlation(cor_final, z_interval, possible_systems):
    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)
    plt.axhline(np.mean(cor_final), color='r', linestyle='--', alpha=0.5)
    plt.axhline(np.std(cor_final) * 3 + np.mean(cor_final), color='yellow', linestyle='--', alpha=0.5)
    plt.axhline(np.std(cor_final) * 2 + np.mean(cor_final), color='#CC7722', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems'])
    plt.show()

def RPF1(peaks_table, synthetic_systems, b):
    tp = sum(1 for system in peaks_table['z'] if any(abs(system - syn)/(1+syn) < 1.665*b/3e5 for syn in synthetic_systems))

    fp = len(peaks_table['z']) - tp
    p = len(synthetic_systems)

    recall = tp / p * 100
    precision = tp / (tp+fp) * 100
    f1 = 2 * tp / (tp + fp + p) * 100

    print('Recall:', recall, '%')
    print('Precision:', precision, '%')
    print('F1 score:', f1, '%')

if __name__ == '__main__':

    spectrum_file = r'D:\Università\terzo anno\Tesi\AC_Correlator\synth_spec.dat'
    
    ## PARAMETERS ##
    # Model parameters
    wav_start = [154.8, 154.8, 155.06]
    wav_end = [155.1, 154.85, 155.1]
    z = 0
    logN = 10
    b = 5
    btur = 0
    ion = 'CIV'

    # Other parameters
    threshold=0.99999
    dz=1e-5
    perc=75
    resol = 45000


    cor_final, z_interval, peaks_table = correlator(spectrum_file, resol, wav_start, wav_end, logN, b, btur, ion, threshold, dz, perc)



    # Completness calculation
    synthetic_systems = [2.55646069, 2.58495365, 2.59680712, 2.60450324, 2.60796068, 2.61049208, 2.64539575, 2.64623804, 2.65191481, 2.66880955, 2.70345038, 2.71184958, 2.71576258, 2.7583452, 2.78713784, 2.81702964, 2.81859274, 2.88002692, 2.89315974, 2.93926941]

    RPF1(peaks_table, synthetic_systems, b)

    plot_correlation(cor_final, z_interval, peaks_table['z'])



    
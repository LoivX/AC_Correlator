import matplotlib.pyplot as plt
from correlator_v2_modfs import correlator
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
    plt.title('Correlation vs Redshift (modfs)')
    plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems'])
    plt.show()

def RPF1(peaks_table, synthetic_systems, b):
    tp = sum(1 for system in peaks_table['z'] if any(abs(system - syn) < 1.665*b/3e5 for syn in synthetic_systems))

    fp = len(peaks_table['z']) - tp
    p = len(synthetic_systems)

    recall = tp / p * 100
    precision = tp / (tp+fp) * 100
    f1 = 2 * tp / (tp + fp + p) * 100

    print('Recall:', recall, '%')
    print('Precision:', precision, '%')
    print('F1 score:', f1, '%')

if __name__ == '__main__':

    spectrum_file = 'testCIV_7_spec.dat'
    
    ## PARAMETERS ##
    # Model parameters
    wav_start = [154.8, 154.8, 155.06]
    wav_end = [155.1, 154.85, 155.1]
    z = 0
    logN = 11.5
    b = 5
    btur = 0
    ion = 'CIV'

    # Other parameters
    threshold=0.999
    dz=1e-5
    perc=75

    #Selection parameters
    height_threshold = 2.5
    z_tolerance = 0.2e-3

    cor_final, z_interval, peaks_table = correlator(spectrum_file, wav_start, wav_end, logN, b, btur, ion, threshold, dz, perc, height_threshold, z_tolerance)
    print(peaks_table)


    # Completness calculation
    synthetic_systems = [2.56835944, 2.46606907, 2.63170324, 2.23751349, 2.18689804, 2.86495142, 2.44759083, 2.47467768, 2.83605921, 2.24308258, 2.33421258, 2.9663927, 2.32172368, 2.21297566, 2.31987066, 2.70090404, 2.371381, 2.22990712, 2.69746521, 2.12421353]
    RPF1(peaks_table, synthetic_systems, b)

    plot_correlation(cor_final, z_interval, peaks_table['z'])



    
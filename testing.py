import matplotlib.pyplot as plt
from correlator_v2 import correlator
import numpy as np
import sys
import os

def plot_correlation(cor_final, z_interval, possible_systems, synthetic_systems, logN, snr, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for system in synthetic_systems:
        plt.axvline(x=system, color='r', alpha=0.3)

    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)
    plt.axhline(np.mean(cor_final), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(np.std(cor_final) * 3 + np.mean(cor_final), color='yellow', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation', 'Mean', r'3$\sigma$', r'2$\sigma$', 'possible Systems'])
    plt.savefig(os.path.join(output_dir, f'cor_{snr}_{logN}.png'))

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
    dz=1e-5
    perc=75
    resol = 45000

    #Selection parameters
    height_threshold = 3.95
    z_tolerance = 8.45e-4

    # Synthetic systems
    synthetic_systems = [2.55646069, 2.58495365, 2.59680712, 2.60450324, 2.60796068, 2.61049208, 2.64539575, 2.64623804, 2.65191481, 2.66880955, 2.70345038, 2.71184958, 2.71576258, 2.7583452, 2.78713784, 2.81702964, 2.81859274, 2.88002692, 2.89315974, 2.93926941
]


    original_stdout = sys.stdout


    for snr in [50, 100, 200]:
        file = open(f"output{snr}.txt", "w")
        sys.stdout = file  # Reindirizza stdout al file
        for i in range(0,25,2):
            spectrum_file = f'test_{i:02d}_{snr}_spec.dat'
            print('\n Processing file:', spectrum_file)
            cor_final, z_interval, peaks_table = correlator(spectrum_file, resol, wav_start, wav_end, logN +0.1*i, b, btur, ion, dz, perc)
            RPF1(peaks_table, synthetic_systems, b)
            plot_correlation(cor_final, z_interval, peaks_table['z'], synthetic_systems, logN +0.1*i, snr, r'D:\Università\terzo anno\Tesi\Immagini\Cor_plots')

            cor_final, z_interval, peaks_table = [], [], []

    sys.stdout = original_stdout
    file.close()




    
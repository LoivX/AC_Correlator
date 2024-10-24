import matplotlib.pyplot as plt
from correlator_v3 import correlator
import numpy as np
import sys
import os

def plot_correlation(cor_final, z_interval, possible_systems, synthetic_systems, logN, snr, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)

    for system in synthetic_systems:
        plt.axvline(x=system, color='r', alpha=0.3)

    plt.axhline(np.std(cor_final) * 3 + np.mean(cor_final), color='orange', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation','Synthetic Systems',r'3$\sigma$', 'possible Systems'])
    plt.savefig(os.path.join(output_dir, f'cor_{snr}_{logN}.png'))

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


if __name__ == '__main__':  
    ## PARAMETERS ##
    # Model parameters
    z = 0
    logN = 10.1
    b = 5
    ion = 'CIV'

    # Other parameters
    dz=1e-5
    resol = 45000

    #Selection parameters
    height_threshold = 3.95
    z_tolerance = 8.45e-4

    # Synthetic systems
    synthetic_systems = [2.51097581, 2.53316622, 2.53765678, 2.55659701, 2.5738651, 2.58102128, 2.60045534, 2.64506485, 2.69327996, 2.72935909, 2.75362322, 2.76679574, 2.78899205, 2.813958, 2.84173642, 2.84820465, 2.95841746, 2.96369678, 2.97805738, 2.99442473]

    original_stdout = sys.stdout


    for snr in [50, 100, 200]:
        file = open(f"output{snr}.txt", "w")
        sys.stdout = file  # Reindirizza stdout al file
        for i in range(0,25,2):
            spectrum_file = f'test_{i:02d}_{snr}_spec.dat'
            print('\n Processing file:', spectrum_file)
            cor_final, z_interval, peaks = correlator(spectrum_file, resol, logN +0.1*i, b, ion, dz)
            RPF1(peaks, synthetic_systems, b)
            plot_correlation(cor_final, z_interval, peaks, synthetic_systems, logN +0.1*i, snr, r'D:\Università\terzo anno\Tesi\Immagini\Cor_plots')


    sys.stdout = original_stdout
    file.close()




    
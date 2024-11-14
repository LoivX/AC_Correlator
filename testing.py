import matplotlib.pyplot as plt
from correlator_v3 import correlator
import numpy as np
import pandas as pd
import sys
import os

def plot_correlation(cor_final, z_interval, possible_systems, synthetic_systems, logN, threshold, snr, j, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(20, 12))
    plt.plot(z_interval, cor_final)

    for system in synthetic_systems:
        plt.axvline(x=system, color='r', alpha=0.3)

    plt.axhline(threshold, color='orange', linestyle='--', alpha=0.5)

    for system in possible_systems:
        plt.axvline(x=system, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Correlation')
    plt.title('Correlation vs Redshift')
    plt.legend(['Correlation','Synthetic Systems',r'5$\sigma$', 'possible Systems'])
    plt.savefig(os.path.join(output_dir, f'cor_{snr}_{logN}_{j}.png'))

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

    return recall, precision, f1


if __name__ == '__main__':  
    ## PARAMETERS ##
    # Model parameters
    z = 0
    logN = 10
    b = 5
    ion = 'CIV'

    # Other parameters
    dz=1e-5
    resol = 45000

    # Synthetic systems
    synthetic_systems = [
    [2.6369262, 2.57356237, 2.92602991, 2.79323457, 2.7021043, 2.78245244, 2.93420251, 2.81751473, 2.79783801, 2.78510194, 2.96656879, 2.71954366, 2.71048923, 2.96576028, 2.89183774, 2.93326703, 2.98745675, 2.67180604, 2.57033334, 2.9407062],
    [2.59429979, 2.79525565, 2.95867643, 2.53222149, 2.65683691, 2.95691983, 2.55052664, 2.70288, 2.84731891, 2.81942482, 2.54051346, 2.87883671, 2.90765031, 2.86052814, 2.81032357, 2.8263793, 2.5226129, 2.71144721, 2.6391155, 2.79253703],
    [2.97303707, 2.82360574, 2.80414695, 2.73462157, 2.86029095, 2.61063977, 2.58535166, 2.83271437, 2.86530508, 2.80258765, 2.72897072, 2.71128734, 2.93599247, 2.84373221, 2.82733022, 2.58431407, 2.77197319, 2.54457673, 2.81716844, 2.5803909],
    [2.84873834, 2.69480584, 2.8731759, 2.79635149, 2.98477235, 2.68851639, 2.53299551, 2.50802848, 2.75877716, 2.90381455, 2.74424552, 2.75937458, 2.50448747, 2.63595098, 2.88899129, 2.8803405, 2.83983751, 2.88091891, 2.64820636, 2.66516385],
    [2.89748522, 2.97201566, 2.66281496, 2.79941963, 2.97766629, 2.50335092, 2.98161489, 2.86759776, 2.80846653, 2.71354047, 2.61241304, 2.91348074, 2.76958925, 2.88101425, 2.59116867, 2.67678011, 2.53114396, 2.85072792, 2.80065789, 2.91048823],
    [2.58482415, 2.78132776, 2.97336979, 2.54021566, 2.93469969, 2.7964731, 2.55121101, 2.78294392, 2.57400471, 2.83489371, 2.51433705, 2.69131543, 2.96209429, 2.90980867, 2.90591623, 2.84438822, 2.61228953, 2.95498534, 2.53613928, 2.73526794],
    [2.74327042, 2.86876995, 2.7133, 2.77970227, 2.65197913, 2.89769091, 2.63326385, 2.91268471, 2.59823378, 2.76760714, 2.70372439, 2.61291718, 2.71966533, 2.79531363, 2.78216186, 2.70031521, 2.57081705, 2.5724643, 2.59482049, 2.6745128],
    [2.78248906, 2.66904665, 2.59384085, 2.65185021, 2.90354634, 2.66118497, 2.55508634, 2.69983193, 2.78247148, 2.82377785, 2.89775937, 2.94589252, 2.80061287, 2.7007699, 2.58969227, 2.54499111, 2.55672979, 2.73740003, 2.51889102, 2.77304277],
    [2.51010875, 2.56900277, 2.6699306, 2.95876107, 2.64944191, 2.71269558, 2.67179586, 2.87617188, 2.72378388, 2.54286037, 2.71101757, 2.80398126, 2.61210868, 2.95993896, 2.74066062, 2.68023957, 2.85034583, 2.52623497, 2.62308641, 2.91049677],
    [2.78899205, 2.53765678, 2.55659701, 2.58102128, 2.72935909, 2.99442473, 2.75362322, 2.96369678, 2.69327996, 2.813958, 2.76679574, 2.53316622, 2.64506485, 2.60045534, 2.95841746, 2.51097581, 2.84173642, 2.84820465, 2.5738651, 2.97805738]
]

    original_stdout = sys.stdout


    for snr in [50, 100, 200]:
        file = open(f"output{snr}.txt", "w")
        sys.stdout = file  # Reindirizza stdout al file
        for i in range(14,4,-1):
            recall = np.zeros(10)
            precision = np.zeros(10)
            f1 = np.zeros(10)

            threshold = correlator(rf'plain_spec_{snr}_spec.dat', resol, logN +0.1*i, b, ion, dz, 1)
            print(f'logN = {logN + 0.1*i},  treshold = {threshold}')

            for j in range(1,10,1):
                spectrum_file = rf'spectra\test_{i:02d}_{snr}_{j}_spec.dat'
                print('Processing file:', spectrum_file)
                cor_final, z_interval, peaks = correlator(spectrum_file, resol, logN +0.1*i, b, ion, dz, threshold)
                recall[j-1], precision[j-1], f1[j-1] = RPF1(peaks, synthetic_systems[j-1], b)
                plot_correlation(cor_final, z_interval, peaks, synthetic_systems[j-1], logN +0.1*i, threshold, snr, j, rf'D:\Università\terzo anno\Tesi\Immagini\v3_final\{snr}\{i:02d}')
                print('\n')

            print('\n \n MEANS')
            print('Recall: ', np.mean(recall))  
            print('Precision: ', np.mean(precision))  
            print('F1 score: ', np.mean(f1))  
            print(' ______________________________________________________________________________ \n \n \n')

            

    sys.stdout = original_stdout
    file.close()




    
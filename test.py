import matplotlib.pyplot as plt
from correlator import correlator
import numpy as np
from astropy.table import Table
import tkinter as tk
from tkinter import filedialog


if __name__ == '__main__':

    def choose_file(title):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(title=title)
        root.destroy()
        return file_path

    plain_spec_file = choose_file('Select the plain spectrum file')
    spectrum_file = choose_file('Select the spectrum file')
    
    # Load spectrum
    spectrum = Table.read(spectrum_file, format='ascii')['x' , 'y']
    plain_spec = Table.read(plain_spec_file, format='ascii')['x' , 'y']
    #spectrum.rename_column('deabs', 'y')

    ## PARAMETERS ##
    # Model parameters
    def get_parameters():
        def on_confirm():
            parameters['logN'] = float(logN_entry.get())
            parameters['b'] = float(b_entry.get())
            parameters['ion'] = ion_entry.get()
            parameters['n'] = float(n_entry.get())
            param_window.destroy()

        param_window = tk.Tk()
        param_window.title("Enter Parameters")

        tk.Label(param_window, text="logN:").grid(row=0, column=0)
        logN_entry = tk.Entry(param_window)
        logN_entry.grid(row=0, column=1)

        tk.Label(param_window, text="b:").grid(row=1, column=0)
        b_entry = tk.Entry(param_window)
        b_entry.grid(row=1, column=1)

        tk.Label(param_window, text="ion:").grid(row=2, column=0)
        ion_entry = tk.Entry(param_window)
        ion_entry.grid(row=2, column=1)

        tk.Label(param_window, text="n (number of sigma):").grid(row=3, column=0)
        n_entry = tk.Entry(param_window)
        n_entry.grid(row=3, column=1)

        confirm_button = tk.Button(param_window, text="Confirm", command=on_confirm)
        confirm_button.grid(row=4, columnspan=2)

        param_window.mainloop()
        return parameters

    parameters = {'logN': None, 'b': None, 'ion': None, 'n': None}
    parameters = get_parameters()
    logN, b, ion, n = parameters['logN'], parameters['b'], parameters['ion'], parameters['n']

    # Other parameters
    dz= 1e-5
    resol = 45000

    
    cor_final, z_interval, peaks = correlator(spectrum, plain_spec, resol, logN, b, ion, dz, n)


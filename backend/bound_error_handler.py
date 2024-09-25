from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import matplotlib.pyplot as plt
import numpy as np
import random


class BoundErrorHandler:
    def __init__(self, input_data,plot_flag = False
                 ):
        
        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.ignore_lowerbound = None
        self.wordlength = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.pysat_thread = None
        self.z3_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.plot_flag = plot_flag
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order_upperbound,
        }

        return input_data_sf
    
    def leak_validator(self, h_res, original_freqx_axis, original_upperbound_lin, original_lowerbound_lin):
        # print("Result plotter called with higher accuracy check")
        
        # Construct fir_coefficients from h_res
        fir_coefficients = np.concatenate((h_res[::-1], h_res[1:]))

        print(fir_coefficients)
        # print("FIR Coefficients in higher accuracy test", fir_coefficients)

        # Compute the FFT of the coefficients at higher resolution
        N = len(original_freqx_axis)*2  # Use the original frequency resolution length for FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N//2]
        
        # Normalize frequencies to range from 0 to 1 for plotting
        omega = frequencies * 2 * np.pi
        normalized_omega = np.linspace(0, 1, len(magnitude_response))

        # Initialize leak detection
        leaks = []
        leaks_mag = []
        continous_leak_count = 0

        # Check for leaks by comparing the FFT result with the 10x accuracy bounds
        for i, mag in enumerate(magnitude_response):
            if mag > original_upperbound_lin[i] + 0.005: 
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append(mag-original_upperbound_lin[i])
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            elif mag < original_lowerbound_lin[i] - 0.005:
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append(np.abs(mag-original_lowerbound_lin[i]))
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            else: continous_flag = False

        # print(f"len(magnitude_response) {len(magnitude_response)}")
        # print(f"len(original_freqx_axis) {len(original_freqx_axis)}")
        # print(f"leaks {leaks}")

        # Plot the input bounds (using the original bounds, which are at higher accuracy)
        self.ax1.scatter(self.freqx_axis, self.upperbound_lin, color='r', s=20, picker=5, label="Upper Bound")
        self.ax1.scatter(self.freqx_axis, self.lowerbound_lin, color='b', s=20, picker=5, label="Lower Bound")

        # Plot the higher accuracy bounds (for validation)
        self.ax2.scatter(original_freqx_axis, original_upperbound_lin, color='r', s=20, picker=5, label="Upper Bound (10x Accuracy)")
        self.ax2.scatter(original_freqx_axis, original_lowerbound_lin, color='b', s=20, picker=5, label="Lower Bound (10x Accuracy)")

        # Plot the magnitude response from the calculated coefficients
        self.ax1.scatter(normalized_omega, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)
        self.ax2.scatter(normalized_omega, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)

        leaks_mag_avg = 0

        # Mark the leaks on the plot
        if leaks:
            leak_indices, leak_values = zip(*leaks)
            leaks_mag = [float(x) for x in leaks_mag]
            leaks_mag_avg = sum(leaks_mag)/len(leaks_mag)

            leak_freqs = [normalized_omega[i] for i in leak_indices]
            self.ax1.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)
            # self.ax2.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)

        self.ax1.set_ylim([-10, 10])
        self.ax2.set_ylim([-10, 10])

        if self.self.plot_flag:
            plt.canvas.draw()
        
        plt.show()


        # Return leaks for further analysis if needed
        return leaks,leaks_mag_avg,continous_leak_count

        






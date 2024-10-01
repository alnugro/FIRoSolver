from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import matplotlib.pyplot as plt
import numpy as np
import math

try:
    from .formulation_pysat import FIRFilterPysat
    from .formulation_z3_pbsat import FIRFilterZ3
    from .formulation_gurobi import FIRFilterGurobi
    from .solver_func import SolverFunc
except:
    from formulation_pysat import FIRFilterPysat
    from formulation_z3_pbsat import FIRFilterZ3
    from formulation_gurobi import FIRFilterGurobi
    from solver_func import SolverFunc



class BoundErrorHandler:
    def __init__(self, input_data
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

        self.original_xdata = None
        self.original_upperbound_lin = None
        self.original_lowerbound_lin = None

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.cutoffs_x = None
        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.solver_accuracy_multiplier = None

        self.patch_multiplier = None



        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        


        self.bound_too_small_flag = False
        self.h_res = None
        self.gain_res = None

        self.half_order = (self.order_upperbound // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2) - 1
        self.ignore_error = 10 ** (-60 / 20) 

        self.plot_flag = True #turn this on to graph result
        if self.plot_flag:
            self.fig, (self.axone, self.axtwo) = plt.subplots(2,1)
            



    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order_upperbound,
        }

        return input_data_sf
    
    def leak_validator(self, h_res, gain_res):

        print("Validation plotter called")
        sf = SolverFunc(self.get_solver_func_dict())
        # Array to store the results of the frequency response computation
        magnitude_response = []
        
        # Recompute the frequency response for each frequency point
        for i, omega in enumerate(self.original_xdata):
            
            term_sum_exprs = 0
            
            # Compute the sum of products of coefficients and the cosine/sine terms with much higher cm accuracy
            for j in range(self.half_order+1):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            magnitude_response.append(np.abs(term_sum_exprs))



        # self.h_res = h_res
        # self.gain_res = gain_res
        # # print("Result plotter called with higher accuracy check")
        
        # # Construct fir_coefficients from h_res
        # fir_coefficients = np.concatenate((h_res[::-1], h_res[1:]))

        # print(fir_coefficients)
        # # print("FIR Coefficients in higher accuracy test", fir_coefficients)

        # # Compute the FFT of the coefficients at higher resolution
        # N = len(self.original_xdata)*2  # Use the original frequency resolution length for FFT
        # frequency_response = np.fft.fft(fir_coefficients, N)
        # frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude response for positive frequencies
        # magnitude_response = np.abs(frequency_response)[:N//2]
        
        # # Normalize frequencies to range from 0 to 1 for plotting
        # omega = frequencies * 2 * np.pi
        # normalized_omega = np.linspace(0, 1, len(magnitude_response))

        # Initialize leak detection
        leaks = []
        leaks_mag = []
        continous_leak_count = 0
        continous_flag = False

        # Check for leaks by comparing the FFT result with the 10x accuracy bounds
        for i, mag in enumerate(magnitude_response):
            if mag - self.ignore_error > self.original_upperbound_lin[i] * gain_res: 
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append((mag-self.original_upperbound_lin[i])/gain_res)
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            elif mag + self.ignore_error < self.original_lowerbound_lin[i] * gain_res:
                if mag - self.ignore_error < self.ignore_lowerbound:
                    continue
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append((mag-self.original_lowerbound_lin[i])/gain_res)
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            else: continous_flag = False

        # print(f"len(magnitude_response) {len(magnitude_response)}")
        # print(f"len(self.original_xdata) {len(self.original_xdata)}")
        # print(f"leaks {leaks}")
        
        if self.plot_flag:
            # Plot the input bounds (using the original bounds, which are at higher accuracy)
            self.axone.scatter(self.xdata, np.array(self.upperbound_lin) * gain_res, color='r', s=20, picker=5, label="Upper Bound")
            self.axone.scatter(self.xdata, np.array(self.lowerbound_lin) * gain_res, color='b', s=20, picker=5, label="Lower Bound")

            # Plot the higher accuracy bounds (for validation)
            self.axtwo.scatter(self.original_xdata, np.array(self.original_upperbound_lin) * gain_res, color='r', s=20, picker=5, label="Upper Bound (10x Accuracy)")
            self.axtwo.scatter(self.original_xdata, np.array(self.original_lowerbound_lin) * gain_res, color='b', s=20, picker=5, label="Lower Bound (10x Accuracy)")

            # Plot the magnitude response from the calculated coefficients
            self.axone.scatter(self.original_xdata, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)
            self.axtwo.scatter(self.original_xdata, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)

        # Mark the leaks on the plot
        if leaks:
            leak_indices, leak_values = zip(*leaks)
            leaks_mag = [float(x) for x in leaks_mag]


            leak_freqs = [self.original_xdata[i] for i in leak_indices]
            if self.plot_flag:
                self.axone.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)
            # self.axtwo.scatter(leak_freqs, leak_values, color='black', s=4, label="Leak Points", zorder=5)

        

        if self.plot_flag:
            self.axone.set_ylim([-10, 10])
            self.axtwo.set_ylim([-10, 10])
            plt.style.use('default')
            plt.show()
    
            
        
    
        # Return leaks for further analysis if needed
        return leaks,leaks_mag
    
    def patch_bound_error(self,leaks,leaks_mag):
        upperbound_cor = np.copy(self.upperbound_lin)
        lowerbound_cor = np.copy(self.lowerbound_lin)
        xdata_cor = np.copy(self.xdata)

        leak_found = np.zeros(len(self.xdata))

        leak_indices, leak_values = zip(*leaks)
        # print((leak_indices))
        # print((leaks_mag))

        for i, leak in enumerate(leak_indices):
            leak_x_value = self.original_xdata[leak]
            if leak_x_value in self.xdata:
                x_indices = int(np.where(self.xdata == leak_x_value)[0][0])
                print(f"leak is inside bound: {leak_x_value} coef accuracy was bad, reducing bounds")
                # print(f"this is inside {x_indices}")
                if upperbound_cor[x_indices] - np.abs(leaks_mag[i]) < lowerbound_cor[x_indices] or lowerbound_cor[x_indices] + np.abs(leaks_mag[i]) > upperbound_cor[x_indices]:
                    #if the bounds become flipped make them to the middle
                    upperbound_cor[x_indices] = (upperbound_cor[x_indices] + lowerbound_cor[x_indices])/2
                    lowerbound_cor[x_indices] = (upperbound_cor[x_indices] + lowerbound_cor[x_indices])/2
                    self.bound_too_small_flag = True
                elif leaks_mag[i] > 0:
                    upperbound_cor[x_indices] -= leaks_mag[i]
                elif leaks_mag[i] < 0:
                    lowerbound_cor[x_indices] -= leaks_mag[i]
                continue

            xdata_value_below= self.find_closest_below(self.xdata, leak_x_value)
            # print(f"this is xdata_index {xdata_value_below}")
            leak_found[int(np.where(self.xdata == xdata_value_below)[0][0])]+=1

        for i, leakf in enumerate(leak_found):
            if leakf > 0:
                xdata_cor = self.interpolate_and_insert(xdata_cor, i, num_values=1 * self.patch_multiplier)
                upperbound_cor = self.interpolate_and_insert(upperbound_cor, i, num_values=1 * self.patch_multiplier)
                lowerbound_cor = self.interpolate_and_insert(lowerbound_cor, i, num_values=1 * self.patch_multiplier)
                

        # print(f"this is leak_found {(leak_found)}")
        # # print(f"this is xdata_cor {(xdata_cor)}")
        # # print(f"this is upperbound_cor {(upperbound_cor)}")
        # # print(f"this is lowerbound_cor {(lowerbound_cor)}")

        # print(f"this is xdata_cor {(len(xdata_cor))}")
        # print(f"this is xdata_cor {(len(self.xdata))}")

        return xdata_cor,upperbound_cor,lowerbound_cor


    def find_closest_below(self, arr, target):
        # Convert to a NumPy array if it's not already
        arr = np.array(arr)
        
        # Filter values that are less than or equal to the target
        filtered_arr = arr[arr <= target]
        
        # If no value is found, return None or NaN
        if filtered_arr.size == 0:
            return None
        
        # Find the closest value
        return filtered_arr[np.abs(filtered_arr - target).argmin()]
    
    def interpolate_and_insert(self, arr, index1, num_values=1):
        index2 = index1 + 1
        # Get values at the two indices
        value1 = arr[index1]
        value2 = arr[index2]
        
        # Create the interpolated values
        interpolated_values = np.linspace(value1, value2, num=num_values + 2)[1:-1]  # Exclude the endpoints
        
        # Insert the interpolated values
        new_arr = np.insert(arr, index2, interpolated_values)
        
        return new_arr

        






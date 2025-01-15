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
        self.half_order = None
        self.intfeastol = None



        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        


        self.bound_too_small_flag = False
        self.h_res = None
        self.gain_res = None


        self.plot_flag = False #turn this on to graph result
        if self.intfeastol == None:
            self.intfeastol = 1e-5 #if not found maybe from test, gurobi feasibility accuracy 1e-5 is set as default

        self.ignore_error = self.intfeastol
        
            



    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
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
            for j in range(self.half_order):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            magnitude_response.append(np.abs(term_sum_exprs))


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
            self.fig, (self.axone, self.axtwo) = plt.subplots(2,1)
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
                    #if the bounds is flipped, interpolate them to the middle
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

        
if __name__ == "__main__":
   # Test inputs
    filter_type = 0
    order_current = 18
    accuracy = 4
    wordlength = 11
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 5
    intW = 1

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 4

    gurobi_thread = 10
    pysat_thread = 0
    z3_thread = 0

    timeout = 0


    passband_error = 0.094922
    stopband_error = 0.094922
    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0

    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point - 1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point - 1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point - 1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point - 1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point - 1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point - 1])

    # Beyond this bound, lowerbound will be ignored
    ignore_lowerbound = -60

    # Linearize the bounds
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
    cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)

    input_data = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'original_xdata': freqx_axis,
        'original_upperbound_lin': upperbound_lin,
        'original_lowerbound_lin': lowerbound_lin,
        'xdata': freqx_axis,
        'upperbound_lin': upperbound_lin,
        'lowerbound_lin': lowerbound_lin,
        'ignore_lowerbound': ignore_lowerbound_lin,
        'cutoffs_x': cutoffs_x,
        'cutoffs_upper_ydata_lin': cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': cutoffs_lower_ydata_lin,
        'wordlength': wordlength,
        'adder_count': adder_count,
        'adder_depth': adder_depth,
        'avail_dsp': avail_dsp,
        'adder_wordlength_ext': adder_wordlength_ext,  # This is extension, not the adder wordlength
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': gurobi_thread,
        'pysat_thread': pysat_thread,
        'z3_thread': z3_thread,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': accuracy,
        'deepsearch': True,
        'patch_multiplier': 1,
        'gurobi_auto_thread': False,
        'seed': 0
    }

    h_res = [208,144,52,-36,-32,0,0,0,0,0]

    h_res = [h / 2**9 for h in h_res]

    print(h_res)

    gain_res = 1
    leak_validator = BoundErrorHandler(input_data)
    leak_validator.plot_flag = True
    leaks,leaks_mag = leak_validator.leak_validator(h_res, gain_res)





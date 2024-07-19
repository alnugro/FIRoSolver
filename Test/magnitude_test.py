import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time


class SolverFunc():
    def __init__(self,filter_type, order):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.gain = 10**1

    def db_to_linear(self,db_arr):
        # Create a mask for NaN values
        nan_mask = np.isnan(db_arr)

        # Apply the conversion to non-NaN values (magnitude)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)

        # Preserve NaN values
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return self.gain
            cm=(2*np.cos(np.pi*omega*m))*self.gain
            return int(cm)
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))


class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_int_res = []
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.freq_upper_lin=0
        self.freq_lower_lin=0
        self.gain = 10**1
        self.ignore_lowerbound_lin = ignore_lowerbound_lin*self.gain


    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.gain) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.gain) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]



        # declaring variables
        h_int = [Int(f'h_int_{i}') for i in range(half_order+1)]

        # Create a Z3 solver instance
        solver = Solver()

        # Create the sum constraints
        for i in range(len(self.freqx_axis)):
            print("upper freq:", self.freq_upper_lin[i])
            print("lower freq:", self.freq_lower_lin[i])
            print("freq:", self.freqx_axis[i])
            term_sum_exprs = 0
            
            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue

            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs += h_int[j] * cm_const
                print("this coef h", j, " is multiplied by ", cm_const)
            solver.add(term_sum_exprs <= self.freq_upper_lin[i])


            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                solver.add(term_sum_exprs >= -self.freq_upper_lin[i])
                continue
            solver.add(term_sum_exprs >= self.freq_lower_lin[i])
            


        for i in range(half_order+1):
                solver.add(h_int[i] <= 2^12)
                solver.add(h_int[i] >= -2^12)

        print("solver Running")
        start_time = time.time()



        if solver.check() == sat:
            print("solver sat")
            model = solver.model()
            for i in range(half_order+1):
                print(f'h_int_{i} = {model[h_int[i]]}')
                self.h_int_res.append(model[h_int[i]].as_long())
                end_time = time.time()

            print(self.h_int_res)
        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

    def plot_result(self, result_coef):
        print("result plotter called")
        fir_coefficients = np.array([])
        for i in range(len(result_coef)):
            fir_coefficients = np.append(fir_coefficients, result_coef[(i+1)*-1])

        for i in range(len(result_coef)-1):
            fir_coefficients = np.append(fir_coefficients, result_coef[i+1])

        print(fir_coefficients)

        print("Fir coef in mp", fir_coefficients)

        # Compute the FFT of the coefficients
        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude and phase response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N//2]

        # Convert magnitude response to dB
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        # print("magdb in mp", magnitude_response_db)

        # Normalize frequencies to range from 0 to 1
        omega= frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])
        # Convert lists to numpy arrays
        freq_upper_lin_array = np.array(self.freq_upper_lin, dtype=np.float64)
        freq_lower_lin_array = np.array(self.freq_lower_lin, dtype=np.float64)

        # Perform element-wise division
        self.freq_upper_lin = (freq_upper_lin_array / self.gain).tolist()
        self.freq_lower_lin = (freq_lower_lin_array / self.gain).tolist()


        #plot input
        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

        # Plot the updated upper_ydata
        self.ax1.plot(normalized_omega, magnitude_response, color='y')

        if self.app:
            self.app.canvas.draw()

    def plot_validation(self):
        print("Validation plotter called")
        half_order = (self.order_current // 2)
        sf = SolverFunc(self.filter_type, self.order_current)
        # Array to store the results of the frequency response computation
        computed_frequency_response = []
        
        # Recompute the frequency response for each frequency point
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            
            # Compute the sum of products of coefficients and the cosine/sine terms
            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, omega)/self.gain
                term_sum_exprs += self.h_int_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))
        
        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()



    

# Test inputs
filter_type = 0
order_upper = 20
accuracy = 4


# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

# Manually set specific values for the elements of freq_upper and freq_lower in dB
freq_upper[30:60] = 2
freq_lower[30:60] = -2

freq_upper[120:130] = -20
freq_lower[120:130] = -1000



#beyond this bound lowerbound will be ignored
ignore_lowerbound_lin = 0.0001

# Create FIRFilter instance
fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin)

# Run solver and plot result
fir_filter.runsolver()
fir_filter.plot_result(fir_filter.h_int_res)
fir_filter.plot_validation()

# Show plot
plt.show()

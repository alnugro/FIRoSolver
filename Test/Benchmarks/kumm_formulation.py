import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time
import re


class SolverFunc():
    def __init__(self,filter_type, order, accuracy):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.coef_accuracy = accuracy

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
                return self.coef_accuracy
            cm=(2*np.cos(np.pi*omega*m))*self.coef_accuracy
            return int(cm)
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))


class FIRFilterKumm:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.freq_upper_lin=0
        self.freq_lower_lin=0
        self.coef_accuracy = 10**3
        self.ignore_lowerbound_lin = ignore_lowerbound
        self.A_M = adder_count
        self.wordlength=wordlength
        self.verbose = False
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)
        self.gain_upperbound= 1.4
        self.gain_lowerbound= 1
        self.gain_res = 0



    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current, self.coef_accuracy)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound_lin, dtype=float)
        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin*self.coef_accuracy



        # declaring variables
        h_int = [Int(f'h_int_{i}') for i in range(half_order+1)]
        gain = Real('gain')

        # Create a Z3 solver instance
        solver = Solver()
        solver.add(gain <= self.gain_upperbound)
        solver.add(gain >= self.gain_lowerbound)

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
                term_sum_exprs += h_int[j]*(2**(-1*self.wordlength)) * cm_const
                print("this coef h", j, " is multiplied by ", cm_const)
            solver.add(term_sum_exprs <= gain*self.freq_upper_lin[i])


            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                solver.add(term_sum_exprs >= gain*-self.freq_upper_lin[i])
                continue
            solver.add(term_sum_exprs >= gain*self.freq_lower_lin[i])
            


        for i in range(half_order+1):
                solver.add(h_int[i] <= 2**self.wordlength)
                solver.add(h_int[i] >= -1*2**self.wordlength)


        #bitshift
        c_a = [Int(f'c_a{a}') for a in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)

        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{a}_{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(c_a[a + 1] == c_sh_sg_a_i[a][0] + c_sh_sg_a_i[a][1])

        c_a_i = [[Int(f'c_a_i{a}_{i}') for i in range(2)] for a in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{a}_{i}_{k}') for k in range(self.A_M + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                for k in range(a + 1):
                    solver.add(Implies(c_a_i_k[a][i][k], c_a_i[a][i] == c_a[k]))
                solver.add(PbEq([(c_a_i_k[a][i][k], 1) for k in range(a + 1)], 1))

        c_sh_a_i = [[Int(f'c_sh_a_i{a}_{i}') for i in range(2)] for a in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{a}_{i}_{s}') for s in range(2 * self.wordlength + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                for s in range(2 * self.wordlength + 1):
                    if s > self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][i][s] == False)
                    if s < self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][0][s] == sh_a_i_s[a][1][s])
                    shift = s - self.wordlength
                    solver.add(Implies(sh_a_i_s[a][i][s], c_sh_a_i[a][i] == (2 ** shift) * c_a_i[a][i]))
                solver.add(PbEq([(sh_a_i_s[a][i][s], 1) for s in range(2 * self.wordlength + 1)], 1))

        sg_a_i = [[Bool(f'sg_a_i{a}_{i}') for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            solver.add(sg_a_i[a][0] + sg_a_i[a][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[a][i], -1 * c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
                solver.add(Implies(Not(sg_a_i[a][i]), c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))

        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{a}_{i}_{s}_{sg}') for sg in range(2)] for s in range(2 * self.wordlength + 1)] for i in range(self.half_order+1)] for a in range(self.A_M + 1)]

        for i in range(self.half_order+1):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        solver.add(Implies(o_a_m_s_sg[a][i][s][sg], (-1 ** sg) * (2 ** shift) * c_a[a] == h_int[i]))
            solver.add(PbEq([(o_a_m_s_sg[a][i][s][sg], 1) for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in range(2)], 1))


        print("solver Running")
        start_time = time.time()

        satifiability = 'unsat'


        if solver.check() == sat:
            satifiability = 'sat'
            print("solver sat")
            model = solver.model()
            for i in range(half_order+1):
                print(f'h_int_{i} = {model[h_int[i]]}')
                h_res=(model[h_int[i]].as_long())*(2**-self.wordlength)
                self.h_res.append(h_res)
                end_time = time.time()
                # Assuming model.eval(gain).as_decimal(5) returns a string representation of a decimal
                gain_decimal_str = model.eval(gain).as_decimal(5)

                # Clean the string by removing any non-numeric characters (except for '.', '-', and digits)
                cleaned_gain_decimal_str = re.sub(r'[^0-9.-]', '', gain_decimal_str)

                try:
                    # Convert the cleaned string to a float and then to np.float64
                    self.gain_res = np.float64(float(cleaned_gain_decimal_str))
                except ValueError:
                    # Handle the case where the string is still not convertible to a float
                    print(f"Error: Could not convert cleaned string '{cleaned_gain_decimal_str}' to float.")
                    self.gain_res = np.nan  # or some other default value

            print(f"gain: {self.gain_res}")
        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        return duration , satifiability
    
    def _print(self, msg):
        if self.verbose:
            print(msg)

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
        self.freq_upper_lin = (freq_upper_lin_array*self.gain_res / self.coef_accuracy).tolist()
        self.freq_lower_lin = (freq_lower_lin_array*self.gain_res / self.coef_accuracy).tolist()


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
        sf = SolverFunc(self.filter_type, self.order_current, self.coef_accuracy)
        # Array to store the results of the frequency response computation
        computed_frequency_response = []
        
        # Recompute the frequency response for each frequency point
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            
            # Compute the sum of products of coefficients and the cosine/sine terms
            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, omega)/self.coef_accuracy
                term_sum_exprs += self.h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))
        
        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()

    

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 16
    accuracy = 1
    adder_count = 8
    wordlength = 10

    space = int(accuracy*order_upper)
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.4*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -20
    freq_lower[upper_half_point:end_point] = -1000




    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterKumm(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()


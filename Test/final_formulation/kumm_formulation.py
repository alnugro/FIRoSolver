import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time


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


class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, app=None):
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
        self.coef_accuracy = 10**10
        self.wordlength = 10
        self.ignore_lowerbound_lin = ignore_lowerbound_lin*self.coef_accuracy
        self.A_M = adder_count
        self.wordlength=wordlength
        self.verbose = True
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)


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
                term_sum_exprs += h_int[j]*(2**(-1*self.wordlength)) * cm_const
                print("this coef h", j, " is multiplied by ", cm_const)
            solver.add(term_sum_exprs <= self.freq_upper_lin[i])


            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                solver.add(term_sum_exprs >= -self.freq_upper_lin[i])
                continue
            solver.add(term_sum_exprs >= self.freq_lower_lin[i])
            


        for i in range(half_order+1):
                solver.add(h_int[i] <= 2**self.wordlength)
                solver.add(h_int[i] >= -1*2**self.wordlength)


        #bitshift
        c_a = [Int(f'c_a{a}') for a in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)

        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(c_a[a + 1] == c_sh_sg_a_i[a][0] + c_sh_sg_a_i[a][1])

        c_a_i = [[Int(f'c_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{a}{i}{k}') for k in range(self.A_M + 1)] for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            for i in range(2):
                for k in range(a + 1):
                    solver.add(Implies(c_a_i_k[a][i][k], c_a_i[a][i] == c_a[k]))
                solver.add(PbEq([(c_a_i_k[a][i][k], 1) for k in range(a + 1)], 1))

        c_sh_a_i = [[Int(f'c_sh_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{a}{i}{s}') for s in range(2 * self.wordlength + 1)] for i in range(2)] for a in range(self.A_M)]

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

        sg_a_i = [[Bool(f'sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]

        for a in range(self.A_M):
            solver.add(sg_a_i[a][0] + sg_a_i[a][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[a][i], -1 * c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
                solver.add(Implies(Not(sg_a_i[a][i]), c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))

        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{a}{i}{s}{sg}') for sg in range(2)] for s in range(2 * self.wordlength + 1)] for i in range(self.half_order+1)] for a in range(self.A_M + 1)]

        for i in range(self.half_order+1):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        solver.add(Implies(o_a_m_s_sg[a][i][s][sg], (-1 ** sg) * (2 ** shift) * c_a[a] == h_int[i]))
            solver.add(PbEq([(o_a_m_s_sg[a][i][s][sg], 1) for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in range(2)], 1))


        print("solver Running")
        start_time = time.time()



        if solver.check() == sat:
            print("solver sat")
            model = solver.model()
            for i in range(half_order+1):
                print(f'h_int_{i} = {model[h_int[i]]}')
                h_res=(model[h_int[i]].as_long())*(2**-self.wordlength)
                self.h_res.append(h_res)
                end_time = time.time()
            self.validate(model, c_a, c_a_i, c_a_i_k, c_sh_a_i, sg_a_i, c_sh_sg_a_i, o_a_m_s_sg, sh_a_i_s, h_int)


            print(self.h_res)
        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

    def validate(self, model, c_a, c_a_i, c_a_i_k, c_sh_a_i, sg_a_i, c_sh_sg_a_i, o_a_m_s_sg, sh_a_i_s, h_int):
        for a in range(0, self.A_M):
            c_a_val = model.eval(c_a[a + 1]).as_long()
            self._print(f"Adder {a + 1}: output of adder {a + 1} is {c_a_val}")

            for i in range(2):
                # Determine the connection
                input_value = 0
                for k in range(a + 1):
                    if is_true(model.eval(c_a_i_k[a][i][k])):
                        input_value = model.eval(c_a[k]).as_long()
                        if i == 0:
                            self._print(f"  Left shifter of Adder {a + 1} is connected to input {k} with a value of {input_value}")
                        else:
                            self._print(f"  Right shifter of Adder {a + 1} is connected to input {k} with a value of {input_value}")

                # Determine the shift
                shift = 0
                for s in range(2 * self.wordlength + 1):
                    if is_true(model.eval(sh_a_i_s[a][i][s])):
                        shift = s - self.wordlength
                        if i == 0:
                            self._print(f"  Left Shifter of Adder {a + 1} shifted by {shift} bits, therefore it's multiplied by {2**shift}")
                        else:
                            self._print(f"  Right Shifter of Adder {a + 1} shifted by {shift} bits, therefore it's multiplied by {2**shift}")

                # Determine the sign
                sign = 0
                if is_true(model.eval(sg_a_i[a][i])):
                    sign = -1
                    if i == 0:
                        self._print(f"  Left Shifter of Adder {a + 1} has negative sign")
                    else:
                        self._print(f"  Right Shifter of Adder {a + 1} has negative sign")
                else:
                    sign = 1
                    if i == 0:
                        self._print(f"  Left Shifter of Adder {a + 1} has positive sign")
                    else:
                        self._print(f"  Right Shifter of Adder {a + 1} has positive sign")

                c_sh_sg_val = model.eval(c_sh_sg_a_i[a][i]).as_long()
                if i == 0:
                    self._print(f"  Left Shifter of Adder {a + 1} end value is {c_sh_sg_val}")
                else:
                    self._print(f"  Right Shifter of Adder {a + 1} end value is {c_sh_sg_val}")

                # Validate left or right shift
                c_sh_sg_val_calc = (2**shift) * sign * input_value
                if c_sh_sg_val_calc != c_sh_sg_val:
                    if i == 0:
                        self._print(f" Validation failed for Left Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                        raise ValueError(f"Validation failed for Left Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                    else:
                        self._print(f" Validation failed for Right Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")
                        raise ValueError(f"Validation failed for Right Shifter of Adder {a + 1}: Expected: {c_sh_sg_val}, Calculated: {c_sh_sg_val_calc}")

                # Validate adder Sums
                c_a_val_calc = sum([model.eval(c_sh_sg_a_i[a][i]).as_long() for i in range(2)])
                if c_a_val != c_a_val_calc:
                    self._print(f"Validation failed for Adder output c_a[{a + 1}] in adder {a + 1}: Model: {c_a_val}, Calculated: {c_a_val_calc}")
                    raise ValueError(f"Validation failed for Adder output c_a[{a + 1}] in adder {a + 1}: Model: {c_a_val}, Calculated: {c_a_val_calc}")

        for i in range(self.half_order+1):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        if is_true(model.eval(o_a_m_s_sg[a][i][s][sg])):
                            c_a_val = model.eval(c_a[a]).as_long()
                            h_int_val = model.eval(h_int[i]).as_long()
                            calculated_value = (-1 ** sg) * (2 ** shift) * c_a_val
                            if sg == 0:
                                sign = 1
                            else:
                                sign = -1
                            self._print(f'Output[{i}] is connected to adder {a + 1} with an adder value of {c_a_val},end result is shifted by {shift}, thus its multiplied by {2**shift} with sign of {sign}')
                            self._print(f'Output[{i}]: Expected: {h_int_val}, Calculated: {calculated_value}')
                            if calculated_value != h_int_val:
                                self._print(f"Validation failed for output[{i}]: Expected: {h_int_val}, Calculated: {calculated_value}")
                                raise ValueError(f"Validation failed for output[{i}]: Expected: {h_int_val}, Calculated: {calculated_value}")
                            
        print("Shifts Validation Completed with no Error")

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
        self.freq_upper_lin = (freq_upper_lin_array / self.coef_accuracy).tolist()
        self.freq_lower_lin = (freq_lower_lin_array / self.coef_accuracy).tolist()


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



    
# Test inputs
filter_type = 0
order_upper = 6
accuracy = 1
adder_count = 5
wordlength = 4

# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

# Manually set specific values for the elements of freq_upper and freq_lower in dB
lower_half_point = int(0.5*(accuracy*order_upper))
upper_half_point = int(0.5*(accuracy*order_upper))
end_point = accuracy*order_upper

freq_upper[0:lower_half_point] = 10
freq_lower[0:lower_half_point] = -5

freq_upper[upper_half_point:end_point] = -10
freq_lower[upper_half_point:end_point] = -1000



#beyond this bound lowerbound will be ignored
ignore_lowerbound_lin = 0.0001

# Create FIRFilter instance
fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)

# Run solver and plot result
fir_filter.runsolver()
fir_filter.plot_result(fir_filter.h_res)
fir_filter.plot_validation()

# Show plot
plt.show()


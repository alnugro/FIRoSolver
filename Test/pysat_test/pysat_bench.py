import numpy as np
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import time
from variable_handler import VariableMapper

class SolverFunc():
    def __init__(self, filter_type, order):
        self.filter_type = filter_type
        self.half_order = (order // 2)
        self.overflow_count = 0

    def db_to_linear(self, db_arr):
        nan_mask = np.isnan(db_arr)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def cm_handler(self, m, omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            return 2 * np.cos(np.pi * omega * m)
        
        if self.filter_type == 1:
            return 2 * np.cos(omega * np.pi * (m + 0.5))

        if self.filter_type == 2:
            return 2 * np.sin(omega * np.pi * (m - 1))

        if self.filter_type == 3:
            return 2 * np.sin(omega * np.pi * (m + 0.5))
        
    def overflow_handler(self, input_value, upper_bound, lower_bound, literal):
        self.overflow_count += 1
        overflow_coef = []
        overflow_lit = []

        if input_value > upper_bound:
            while input_value > upper_bound:
                overflow_coef.append(upper_bound)
                overflow_lit.append(literal)
                input_value -= upper_bound
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
        
        elif input_value < lower_bound:
            while input_value < lower_bound:
                overflow_coef.append(lower_bound)
                overflow_lit.append(literal)
                input_value -= lower_bound
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
        
        else:
            overflow_coef.append(input_value)
            overflow_lit.append(literal)
            print("something weird happens on overflow handler")

        return [overflow_lit, overflow_coef]

class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.N = adder_count

        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0

        self.coef_accuracy = 3
        self.wordlength = 15
        self.ih = 2
        self.fh = self.wordlength - self.ih

        self.gain_wordlength = 6 
        self.ig = 3
        self.fg = self.gain_wordlength - self.ig

        self.gain_upperbound = 1.4
        self.gain_lowerbound = 1.3
        self.gain_bound_accuracy = 2  # 2 floating points

        self.ignore_lowerbound_lin = ignore_lowerbound_lin * (10 ** self.coef_accuracy) * (2 ** self.fh)

    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)
        var_mapper = VariableMapper(half_order, self.wordlength, self.gain_wordlength, self.N)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)

        self.freq_upper_lin = [int((sf.db_to_linear(f)) * (10 ** self.coef_accuracy) * (2 ** self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * (10 ** self.coef_accuracy) * (2 ** self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]

        solver = Solver()

        gain_coeffs = []
        gain_literals = []

        self.gain_upperbound_int = int(self.gain_upperbound * 2 ** self.fg * (10 ** self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound * 2 ** self.fg * (10 ** self.gain_bound_accuracy))

        for g in range(self.gain_wordlength):
            gain_coeffs.append((2 ** g) * (10 ** self.gain_bound_accuracy))
            gain_literals.append(v2i(('gain', g)))
            print(v2i(('gain', g)))

        print("gain coef: ",gain_coeffs)
        inverted_gain_coeffs = [-ga for ga in gain_coeffs]
        print("inverted gain coef: ",inverted_gain_coeffs)
        inverted_lowerbound = -1 *self.gain_lowerbound_int


        print("upperbound: ", self.gain_upperbound_int)
        print("lowerbound: ", inverted_lowerbound)


        solver.add_atmost(lits=gain_literals, k=self.gain_upperbound_int, weights=gain_coeffs)
        solver.add_atmost(lits=gain_literals, k=inverted_lowerbound, weights=inverted_gain_coeffs)

        
        is_sat = solver.solve()


        if is_sat:
            model = solver.get_model()
            bool_value = model[1]
            print(bool_value)

            print("SAT")
            print("Model:", model)
            gain_coef = 0
            for g in range(self.gain_wordlength):
                var_index = v2i(('gain', g)) -1
                bool_value = model[var_index] > 0  # Convert to boolean
                print(f"gain{g}= ",var_index ,bool_value)

                if g < self.fg:
                    gain_coef += 2 ** -(self.fg - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fg) * bool_value

            self.gain_res = gain_coef
            print("gain Coeffs: ", self.gain_res)
        else:
            print("UNSAT")

        

# Test inputs
filter_type = 0
order_upper = 14
accuracy = 1
adder_count = 5
wordlength = 4

# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, accuracy * order_upper)
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

lower_half_point = int(0.5 * (accuracy * order_upper))
upper_half_point = int(0.5 * (accuracy * order_upper))
end_point = accuracy * order_upper

freq_upper[0:lower_half_point] = 10
freq_lower[0:lower_half_point] = -5

freq_upper[upper_half_point:end_point] = -10
freq_lower[upper_half_point:end_point] = -1000

ignore_lowerbound_lin = 0.0001

fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)

fir_filter.runsolver()
fir_filter.plot_result(fir_filter.h_res)
fir_filter.plot_validation()

plt.show()

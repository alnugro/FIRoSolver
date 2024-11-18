import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import re

class SolverFunc():
    def __init__(self, filter_type, order, accuracy):
        self.filter_type = filter_type
        self.half_order = (order // 2)
        self.coef_accuracy = accuracy

    def db_to_linear(self, db_arr):
        nan_mask = np.isnan(db_arr)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)
        linear_array[nan_mask] = np.nan
        return linear_array

    def cm_handler(self, m, omega):
        if self.filter_type == 0:
            if m == 0:
                return self.coef_accuracy
            cm = (2 * np.cos(np.pi * omega * m)) * self.coef_accuracy
            return int(cm)

        # Ignore the rest; it's for later use if type 1 works
        if self.filter_type == 1:
            return 2 * np.cos(omega * np.pi * (m + 0.5))

        if self.filter_type == 2:
            return 2 * np.sin(omega * np.pi * (m - 1))

        if self.filter_type == 3:
            return 2 * np.sin(omega * np.pi * (m + 0.5))

class FIRFilterKumm:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength,timeout = None, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0
        self.coef_accuracy = 10 ** 4
        self.ignore_lowerbound_lin = ignore_lowerbound
        self.A_M = adder_count
        self.wordlength = wordlength
        self.verbose = False
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)
        self.gain_upperbound = 1.4
        self.gain_lowerbound = 1
        self.gain_res = 0
        self.timeout = timeout


    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        print("Solver called")
        sf = SolverFunc(self.filter_type, self.order_current, self.coef_accuracy)
        print("Filter order:", self.order_current)
        print("Ignore lower than:", self.ignore_lowerbound_lin)

        # Linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound_lin, dtype=float)
        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin * self.coef_accuracy

        # Create a Gurobi model
        m = gp.Model('FIRFilter')
        if self.timeout != None:
            m.Params.TimeLimit = self.timeout

        # Declare variables
        h_int = m.addVars(half_order + 1, lb=-2 ** self.wordlength, ub=2 ** self.wordlength, vtype=GRB.INTEGER, name='h_int')
        gain = m.addVar(lb=self.gain_lowerbound, ub=self.gain_upperbound, vtype=GRB.CONTINUOUS, name='gain')

        # Gain constraints
        m.addConstr(gain <= self.gain_upperbound)
        m.addConstr(gain >= self.gain_lowerbound)

        # Frequency constraints
        for i in range(len(self.freqx_axis)):
            term_sum_exprs = gp.LinExpr()
            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue
            for j in range(half_order + 1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs += h_int[j] * (2 ** (-1 * self.wordlength)) * cm_const
            m.addConstr(term_sum_exprs <= gain * self.freq_upper_lin[i])

            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                m.addConstr(term_sum_exprs >= gain * -self.freq_upper_lin[i])
                continue
            m.addConstr(term_sum_exprs >= gain * self.freq_lower_lin[i])

        # Variables for adder network
        c_a = {}
        c_a[0] = 1  # c_a[0] is constant
        for a in range(1, self.A_M + 2):
            c_a[a] = m.addVar(vtype=GRB.INTEGER, name=f'c_a{a}')

        c_sh_sg_a_i = {}
        for a in range(self.A_M):
            for i in range(2):
                c_sh_sg_a_i[a, i] = m.addVar(vtype=GRB.INTEGER, name=f'c_sh_sg_a_i{a}_{i}')

        for a in range(self.A_M):
            m.addConstr(c_a[a + 1] == c_sh_sg_a_i[a, 0] + c_sh_sg_a_i[a, 1])

        c_a_i = {}
        for a in range(self.A_M):
            for i in range(2):
                c_a_i[a, i] = m.addVar(vtype=GRB.INTEGER, name=f'c_a_i{a}_{i}')

        c_a_i_k = {}
        for a in range(self.A_M):
            for i in range(2):
                for k in range(a + 1):
                    c_a_i_k[a, i, k] = m.addVar(vtype=GRB.BINARY, name=f'c_a_i_k{a}_{i}_{k}')

        # Constraints for c_a_i_k
        for a in range(self.A_M):
            for i in range(2):
                m.addConstr(gp.quicksum(c_a_i_k[a, i, k] for k in range(a + 1)) == 1)
                m.addConstr(c_a_i[a, i] == gp.quicksum(c_a[k] * c_a_i_k[a, i, k] for k in range(a + 1)))

        c_sh_a_i = {}
        sh_a_i_s = {}
        for a in range(self.A_M):
            for i in range(2):
                c_sh_a_i[a, i] = m.addVar(vtype=GRB.INTEGER, name=f'c_sh_a_i{a}_{i}')
                for s in range(2 * self.wordlength + 1):
                    sh_a_i_s[a, i, s] = m.addVar(vtype=GRB.BINARY, name=f'sh_a_i_s{a}_{i}_{s}')

        # Constraints for sh_a_i_s and c_sh_a_i
        for a in range(self.A_M):
            for i in range(2):
                m.addConstr(gp.quicksum(sh_a_i_s[a, i, s] for s in range(2 * self.wordlength + 1)) == 1)
                shifts = []
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    # Skipping invalid shifts
                    if s > self.wordlength and i == 0:
                        m.addConstr(sh_a_i_s[a, i, s] == 0)
                    if s < self.wordlength and i == 0:
                        m.addConstr(sh_a_i_s[a, 0, s] == sh_a_i_s[a, 1, s])
                    # Create indicator constraints
                    m.addGenConstrIndicator(sh_a_i_s[a, i, s], True, c_sh_a_i[a, i] == (2 ** shift) * c_a_i[a, i])

        # Sign variables
        sg_a_i = {}
        for a in range(self.A_M):
            for i in range(2):
                sg_a_i[a, i] = m.addVar(vtype=GRB.BINARY, name=f'sg_a_i{a}_{i}')

        # Constraints for sg_a_i
        for a in range(self.A_M):
            m.addConstr(sg_a_i[a, 0] + sg_a_i[a, 1] <= 1)
            for i in range(2):
                m.addGenConstrIndicator(sg_a_i[a, i], True, c_sh_sg_a_i[a, i] == -1 * c_sh_a_i[a, i])
                m.addGenConstrIndicator(sg_a_i[a, i], False, c_sh_sg_a_i[a, i] == c_sh_a_i[a, i])

        # Output variables
        o_a_m_s_sg = {}
        for a in range(self.A_M + 1):
            for i in range(half_order + 1):
                for s in range(2 * self.wordlength + 1):
                    for sg in range(2):
                        o_a_m_s_sg[a, i, s, sg] = m.addVar(vtype=GRB.BINARY, name=f'o_a_m_s_sg{a}_{i}_{s}_{sg}')

        # Constraints for o_a_m_s_sg and h_int
        for i in range(half_order + 1):
            m.addConstr(gp.quicksum(o_a_m_s_sg[a, i, s, sg] for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in [0, 1]) == 1)
            h_int_contrib = []
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in [0, 1]:
                        sign = (-1) ** sg
                        coef = sign * (2 ** shift)
                        h_int_contrib_var = m.addVar(vtype=GRB.INTEGER, name=f'h_int_contrib_{a}_{i}_{s}_{sg}')
                        m.addConstr(h_int_contrib_var == coef * c_a[a] * o_a_m_s_sg[a, i, s, sg])
                        h_int_contrib.append(h_int_contrib_var)
            m.addConstr(h_int[i] == gp.quicksum(h_int_contrib))

        # Objective (if any). Since the original code does not specify an objective, we can set it to zero
        m.setObjective(0, GRB.MINIMIZE)

        # Optimize the model
        print("Solver running")
        start_time = time.time()
        m.optimize()
        end_time = time.time()

        # Check if the model is optimal
        if m.status == GRB.OPTIMAL:
            print("Solver status: Optimal")
            for i in range(half_order + 1):
                h_res = h_int[i].X * (2 ** -self.wordlength)
                print(f'h_int_{i} = {h_int[i].X}')
                self.h_res.append(h_res)
            self.gain_res = gain.X
            print(f"gain: {self.gain_res}")
            satifiability = 'sat'
        else:
            print("No optimal solution found.")
            self.gain_res = np.nan  # Assign NaN if no solution
            satifiability = 'unsat'

        print("Solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        return duration, satifiability

    def plot_result(self, result_coef):
        print("Result plotter called")
        fir_coefficients = np.array([])
        for i in range(len(result_coef)):
            fir_coefficients = np.append(fir_coefficients, result_coef[(i + 1) * -1])

        for i in range(len(result_coef) - 1):
            fir_coefficients = np.append(fir_coefficients, result_coef[i + 1])

        print(fir_coefficients)
        print("FIR coefficients in mp", fir_coefficients)

        # Compute the FFT of the coefficients
        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N // 2]  # Extract positive frequencies up to Nyquist

        # Compute the magnitude and phase response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N // 2]

        # Convert magnitude response to dB
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        # Normalize frequencies to range from 0 to 1
        omega = frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])

        # Convert lists to numpy arrays
        freq_upper_lin_array = np.array(self.freq_upper_lin, dtype=np.float64)
        freq_lower_lin_array = np.array(self.freq_lower_lin, dtype=np.float64)

        # Perform element-wise division
        self.freq_upper_lin = (freq_upper_lin_array * self.gain_res / self.coef_accuracy).tolist()
        self.freq_lower_lin = (freq_lower_lin_array * self.gain_res / self.coef_accuracy).tolist()

        # Plot input
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
        computed_frequency_response = []
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            for j in range(half_order + 1):
                cm_const = sf.cm_handler(j, omega) / self.coef_accuracy
                term_sum_exprs += self.h_res[j] * cm_const
            computed_frequency_response.append(np.abs(term_sum_exprs))
        self.ax1.plot([x / 1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        self.ax2.set_ylim(-10, 10)
        if self.app:
            self.app.canvas.draw()



if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 6
    accuracy = 3
    adder_count = 5
    wordlength = 4

    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, accuracy * order_upper)
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * (accuracy * order_upper))
    upper_half_point = int(0.5 * (accuracy * order_upper))
    end_point = accuracy * order_upper

    freq_upper[0:lower_half_point] = 10
    freq_lower[0:lower_half_point] = -5

    freq_upper[upper_half_point:end_point] = -10
    freq_lower[upper_half_point:end_point] = -1000

    # Beyond this bound, lower bound will be ignored
    ignore_lowerbound = -30

    # Create FIRFilter instance
    fir_filter = FIRFilterKumm(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()

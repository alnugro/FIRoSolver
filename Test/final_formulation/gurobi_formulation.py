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
        
        if self.filter_type == 1:
            return 2 * np.cos(omega * np.pi * (m + 0.5))

        if self.filter_type == 2:
            return 2 * np.sin(omega * np.pi * (m - 1))

        if self.filter_type == 3:
            return 2 * np.sin(omega * np.pi * (m + 0.5))


class FIRFilter:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength, app=None):
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
        self.coef_accuracy = 10 ** 3
        self.ignore_lowerbound_lin = ignore_lowerbound
        self.A_M = adder_count
        self.wordlength = wordlength
        self.verbose = False
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)
        self.gain_upperbound = 1.4
        self.gain_lowerbound = 1
        self.gain_res = 0

    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current, self.coef_accuracy)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound_lin, dtype=float)
        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin * self.coef_accuracy

        # Create a Gurobi model instance
        model = gp.Model("FIRFilterOptimization")

        # Declaring variables
        h_int = model.addVars(half_order+1, vtype=GRB.INTEGER, lb=-2**self.wordlength, ub=2**self.wordlength, name="h_int")
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")

        # Set objective (if necessary, for now, let's assume it's a feasibility problem)
        model.setObjective(0, GRB.MINIMIZE)

        # Create the sum constraints
        for i in range(len(self.freqx_axis)):
            print("upper freq:", self.freq_upper_lin[i])
            print("lower freq:", self.freq_lower_lin[i])
            print("freq:", self.freqx_axis[i])
            term_sum_exprs = gp.LinExpr()
            
            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue

            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs += h_int[j] * (2**(-1*self.wordlength)) * cm_const
                print("this coef h", j, " is multiplied by ", cm_const)

            model.addConstr(term_sum_exprs <= gain * self.freq_upper_lin[i])

            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                model.addConstr(term_sum_exprs >= gain * -self.freq_upper_lin[i])
                continue
            model.addConstr(term_sum_exprs >= gain * self.freq_lower_lin[i])

        # # Adder tree and bitshift logic in Gurobi
        # c_a = model.addVars(self.A_M + 1, vtype=GRB.INTEGER, name="c_a")
        # c_a[0].lb = 1
        # c_a[0].ub = 1

        # c_sh_sg_a_i = model.addVars(self.A_M, 2, vtype=GRB.INTEGER, name="c_sh_sg_a_i")
        # c_a_i = model.addVars(self.A_M, 2, vtype=GRB.INTEGER, name="c_a_i")
        # c_a_i_k = model.addVars(self.A_M, 2, self.A_M + 1, vtype=GRB.BINARY, name="c_a_i_k")

        # for a in range(self.A_M):
        #     model.addConstr(c_a[a + 1] == c_sh_sg_a_i[a, 0] + c_sh_sg_a_i[a, 1])

        # for a in range(self.A_M):
        #     for i in range(2):
        #         model.addConstr(gp.quicksum(c_a_i_k[a, i, k] for k in range(a + 1)) == 1)
        #         for k in range(a + 1):
        #             model.addConstr((c_a_i_k[a, i, k] == 1) >> (c_a_i[a, i] == c_a[k]))

        # c_sh_a_i = model.addVars(self.A_M, 2, vtype=GRB.INTEGER, name="c_sh_a_i")
        # sh_a_i_s = model.addVars(self.A_M, 2, 2 * self.wordlength + 1, vtype=GRB.BINARY, name="sh_a_i_s")

        # for a in range(self.A_M):
        #     for i in range(2):
        #         for s in range(2 * self.wordlength + 1):
        #             shift = s - self.wordlength
        #             model.addConstr((sh_a_i_s[a, i, s] == 1) >> (c_sh_a_i[a, i] == (2 ** shift) * c_a_i[a, i]))

        #         model.addConstr(gp.quicksum(sh_a_i_s[a, i, s] for s in range(2 * self.wordlength + 1)) == 1)

        # sg_a_i = model.addVars(self.A_M, 2, vtype=GRB.BINARY, name="sg_a_i")

        # for a in range(self.A_M):
        #     model.addConstr(sg_a_i[a, 0] + sg_a_i[a, 1] <= 1)
        #     for i in range(2):
        #         model.addConstr((sg_a_i[a, i] == 1) >> (-1 * c_sh_a_i[a, i] == c_sh_sg_a_i[a, i]))
        #         model.addConstr((sg_a_i[a, i] == 0) >> (c_sh_a_i[a, i] == c_sh_sg_a_i[a, i]))

        # o_a_m_s_sg = model.addVars(self.A_M + 1, self.half_order+1, 2 * self.wordlength + 1, 2, vtype=GRB.BINARY, name="o_a_m_s_sg")

        # for i in range(self.half_order+1):
        #     for a in range(self.A_M + 1):
        #         for s in range(2 * self.wordlength + 1):
        #             shift = s - self.wordlength
        #             for sg in range(2):
        #                 model.addConstr((o_a_m_s_sg[a, i, s, sg] == 1) >> ((-1 ** sg) * (2 ** shift) * c_a[a] == h_int[i]))

        #         model.addConstr(gp.quicksum(o_a_m_s_sg[a, i, s, sg] for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in range(2)) == 1)

        # print("solver Running")
        start_time = time.time()

        # Optimize the model
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("solver optimal")
            for i in range(half_order+1):
                h_res = h_int[i].x * (2**-self.wordlength)
                self.h_res.append(h_res)
            self.gain_res = gain.x
            print(f"gain: {self.gain_res}")
        else:
            print("No optimal solution found")
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

        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

        magnitude_response = np.abs(frequency_response)[:N//2]
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        omega = frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])

        freq_upper_lin_array = np.array(self.freq_upper_lin, dtype=np.float64)
        freq_lower_lin_array = np.array(self.freq_lower_lin, dtype=np.float64)

        self.freq_upper_lin = (freq_upper_lin_array * self.gain_res / self.coef_accuracy).tolist()
        self.freq_lower_lin = (freq_lower_lin_array * self.gain_res / self.coef_accuracy).tolist()

        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

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

            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, omega) / self.coef_accuracy
                term_sum_exprs += self.h_res[j] * cm_const

            computed_frequency_response.append(np.abs(term_sum_exprs))

        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        self.ax2.set_ylim(-10, 10)

        if self.app:
            self.app.canvas.draw()


# Test inputs
filter_type = 0
order_upper = 50
accuracy = 15
adder_count = 10
wordlength = 12

freqx_axis = np.linspace(0, 1, accuracy * order_upper)  # according to Mr. Kumms paper
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

lower_half_point = int(0.4 * (accuracy * order_upper))
upper_half_point = int(0.6 * (accuracy * order_upper))
end_point = accuracy * order_upper

freq_upper[0:lower_half_point] = 3
freq_lower[0:lower_half_point] = 0

freq_upper[upper_half_point:end_point] = -30
freq_lower[upper_half_point:end_point] = -1000

ignore_lowerbound = -30

fir_filter = FIRFilter(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

fir_filter.runsolver()
fir_filter.plot_result(fir_filter.h_res)
fir_filter.plot_validation()

plt.show()

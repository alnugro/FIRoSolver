import numpy as np
from gurobipy import Model, GRB, quicksum

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
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength,timeout = None, app=None):
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
        self.coef_accuracy = 10**4
        self.ignore_lowerbound_lin = ignore_lowerbound
        self.A_M = adder_count
        self.wordlength=wordlength
        self.verbose = False
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)
        self.gain_upperbound= 4
        self.gain_lowerbound= 1
        self.gain_res = 0
        self.timeout = timeout




    def runsolver(self):
        self.freq_upper_lin= None
        self.freq_lower_lin=None

        self.order_current = int(self.order_upper)
        half_order = self.order_current // 2

        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current, self.coef_accuracy)

        
        # Linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]

        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound_lin, dtype=float)
        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin * self.coef_accuracy

        print("Running Gurobi with the following parameters:")
        print(f"thread: 0")
        print(f"filter_type: {self.filter_type}")
        print(f"order_current: {self.order_current}")
        print(f"freqx_axis: {self.freqx_axis}")
        print(f"upperbound_lin: {self.freq_upper_lin}")
        print(f"lowerbound_lin: {self.freq_lower_lin}")
        print(f"ignore_lowerbound: {self.ignore_lowerbound_lin}")
        print(f"gain_upperbound: {self.gain_upperbound}")
        print(f"gain_lowerbound: {self.gain_lowerbound}")
        print(f"wordlength: {self.wordlength}")


        # Create a Gurobi model instance
        model = Model()

        # Declare variables
        # h_int = model.addVars(half_order + 1, vtype=GRB.INTEGER, name='h_int')
        h_int = model.addVars(half_order + 1, lb=4 * -1 * 2 ** self.wordlength, ub=4 * 2 ** self.wordlength,vtype=GRB.INTEGER, name='h_int')

        gain = model.addVar(lb=self.gain_lowerbound, ub=self.gain_upperbound, vtype=GRB.CONTINUOUS, name='gain')

        model.setObjective(0, GRB.MINIMIZE)

        if self.timeout != None:
            model.Params.TimeLimit = self.timeout

        # Set the gain constraints
        model.addConstr(gain <= self.gain_upperbound)
        model.addConstr(gain >= self.gain_lowerbound)

        # model.Params.NumericFocus = 3  # Improves numerical stability
        # model.Params.FeasibilityTol = 1e-6  # Adjust feasibility tolerance


        # # Constraints on h_int variables
        # for i in range(half_order + 1):
        #     model.addConstr(h_int[i] <= 4 * 2 ** self.wordlength)
        #     model.addConstr(h_int[i] >= 4 * -1 * 2 ** self.wordlength)

        # Per-frequency constraints
        for i in range(len(self.freqx_axis)):
            print("upper freq:", self.freq_upper_lin[i])
            print("lower freq:", self.freq_lower_lin[i])
            print("freq:", self.freqx_axis[i])
            term_sum_exprs = 0

            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue

            cm_consts = []
            for j in range(half_order + 1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                cm_consts.append(cm_const)
                term_sum_exprs += h_int[j] * (2 ** (-1 * self.wordlength)) * cm_const
                # print("this coef h", j, " is multiplied by ", cm_const)

            # Build the term sum expression
            # term_sum_exprs = quicksum(h_int[j] * (2 ** (-1 * self.wordlength)) * cm_consts[j] for j in range(half_order + 1))

            # Upper bound constraint
            model.addConstr(term_sum_exprs <= gain * self.freq_upper_lin[i])

            # Lower bound constraint with ignore lower bound
            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                model.addConstr(term_sum_exprs >= gain * -self.freq_upper_lin[i])
                continue
            else:
                model.addConstr(term_sum_exprs >= gain * self.freq_lower_lin[i])

        # Bit-shifting and sign variables
        A_M = self.A_M
        wordlength = self.wordlength

        # c_a variables
        c_a = model.addVars(A_M + 1, vtype=GRB.INTEGER, name='c_a')
        model.addConstr(c_a[0] == 1)

        # c_sh_sg_a_i variables
        c_sh_sg_a_i = model.addVars(A_M, 2, vtype=GRB.INTEGER, name='c_sh_sg_a_i')

        # c_a variables relationship
        for a in range(A_M):
            model.addConstr(c_a[a + 1] == c_sh_sg_a_i[a, 0] + c_sh_sg_a_i[a, 1])

        # c_a_i and c_a_i_k variables
        c_a_i = model.addVars(A_M, 2, vtype=GRB.INTEGER, name='c_a_i')
        c_a_i_k = model.addVars(A_M, 2, A_M + 1, vtype=GRB.BINARY, name='c_a_i_k')

        for a in range(A_M):
            for i in range(2):
                # Sum over k to select c_a[k]
                model.addConstr(c_a_i[a, i] == quicksum(c_a[k] * c_a_i_k[a, i, k] for k in range(a + 1)))
                # Exactly one k should be selected
                model.addConstr(quicksum(c_a_i_k[a, i, k] for k in range(a + 1)) == 1)

        # c_sh_a_i and sh_a_i_s variables
        max_shift = 2 * wordlength + 1
        c_sh_a_i = model.addVars(A_M, 2, vtype=GRB.INTEGER, name='c_sh_a_i')
        sh_a_i_s = model.addVars(A_M, 2, max_shift, vtype=GRB.BINARY, name='sh_a_i_s')

        for a in range(A_M):
            for i in range(2):
                # Build the shift expression
                shift_expr = quicksum((2 ** (s - wordlength)) * c_a_i[a, i] * sh_a_i_s[a, i, s] for s in range(max_shift))
                model.addConstr(c_sh_a_i[a, i] == shift_expr)
                # Exactly one shift should be selected
                model.addConstr(quicksum(sh_a_i_s[a, i, s] for s in range(max_shift)) == 1)
                # Apply specific constraints based on your Z3 code
                for s in range(max_shift):
                    if s > wordlength and i == 0:
                        model.addConstr(sh_a_i_s[a, i, s] == 0)
                    if s < wordlength and i == 0:
                        model.addConstr(sh_a_i_s[a, 0, s] == sh_a_i_s[a, 1, s])

        # sg_a_i variables
        sg_a_i = model.addVars(A_M, 2, vtype=GRB.BINARY, name='sg_a_i')

        for a in range(A_M):
            # Only one sg_a_i can be 1
            model.addConstr(sg_a_i[a, 0] + sg_a_i[a, 1] <= 1)
            for i in range(2):
                # c_sh_sg_a_i relation with sign
                model.addConstr(c_sh_sg_a_i[a, i] == c_sh_a_i[a, i] * (1 - 2 * sg_a_i[a, i]))

        # o_a_m_s_sg variables
        o_a_m_s_sg = model.addVars(A_M + 1, half_order + 1, max_shift, 2, vtype=GRB.BINARY, name='o_a_m_s_sg')

        for i in range(half_order + 1):
            # Exactly one combination per h_int[i]
            model.addConstr(quicksum(o_a_m_s_sg[a, i, s, sg] for a in range(A_M + 1) for s in range(max_shift) for sg in range(2)) == 1)
            # h_int relationship
            expr = quicksum(((-1) ** sg) * (2 ** (s - wordlength)) * c_a[a] * o_a_m_s_sg[a, i, s, sg] for a in range(A_M + 1) for s in range(max_shift) for sg in range(2))
            model.addConstr(h_int[i] == expr)

        print("solver Running")
        start_time = time.time()

        # # Set Gurobi parameters if needed
        # model.Params.OutputFlag = 1  # Enable solver output

        # Optimize the model
        model.optimize()


        if model.Status == GRB.TIME_LIMIT:
            end_time = time.time()

            satifiability = 'Timeout'

        elif model.status == GRB.OPTIMAL:
            end_time = time.time()
            print("solver sat")
            self.h_res = []
            for i in range(half_order + 1):
                h_value = h_int[i].X * (2 ** -self.wordlength)
                self.h_res.append(h_value)
                print(f'h_int_{i} = {h_int[i].X}')
            self.gain_res = gain.X
            print(f"gain: {self.gain_res}")
            satifiability = 'sat'

        else:
            end_time = time.time()
            print("Unsatisfiable or no optimal solution found")
            satifiability = 'unsat'

    

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        model.dispose()  # Dispose of the model
        del model

        return duration, satifiability
    
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
    order_upper = 20
    accuracy = 2
    adder_count = 10
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

    freq_upper[0:lower_half_point] = 3
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


import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB



class SolverFunc():
    def __init__(self,filter_type, order):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.overflow_count = 0

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
                return 1
            cm=(2*np.cos(np.pi*omega*m))
            return cm
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))




class FIRFilterGurobi:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength,timeout = None, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.max_adder = adder_count

        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        self.freq_upper_lin=0
        self.freq_lower_lin=0

        self.coef_accuracy = 3
        self.intW = 4
        self.fracW = self.wordlength - self.intW

        
        self.gain_wordlength=6 #9 bits wordlength for gain
        self.gain_intW = 3
        self.gain_fracW =  self.gain_wordlength - self.gain_intW

        self.gain_upperbound= 1.4
        self.gain_lowerbound= 1
        self.gain_bound_accuracy = 2 #2 floating points


        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = 0
        self.avail_dsp = 0
        self.adder_wordlength = self.wordlength + 2
        self.result_model = {}

        self.timeout = timeout




    def runsolver(self):
        self.freq_upper_lin= None
        self.freq_lower_lin=None

        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound)
        # linearize the bounds
        self.freq_upper_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f))*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound, dtype=float)

        self.ignore_lowerbound_lin = sf.db_to_linear(self.ignore_lowerbound_np)
        self.ignore_lowerbound_lin = self.ignore_lowerbound_lin*(10**self.coef_accuracy)*(2**self.fracW)
        
        model = gp.Model("example_model")


        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")

        model.setObjective(0, GRB.MINIMIZE)

        if self.timeout != None:
            model.Params.TimeLimit = self.timeout



        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10** self.coef_accuracy)*(-1*(2**w)))
                    else: cm_word_prod= int(cm*(10** self.coef_accuracy)*(2**w))
                    h_sum_temp += h[m][w]*cm_word_prod

            model.update()
            print(f"sum temp is{h_sum_temp}")
            upperbound_constraint = model.addConstr(h_sum_temp <= gain*self.freq_upper_lin[omega], "upperbound_constraint")
            
            
            if self.freq_lower_lin[omega] < self.ignore_lowerbound_lin:
                lowerbound_constraint = model.addConstr(h_sum_temp >= gain*-self.freq_upper_lin[omega], "lowerbound_constraint")
            else:
                lowerbound_constraint = model.addConstr(h_sum_temp >= gain*self.freq_lower_lin[omega], "lowerbound_constraint")



        # Bitshift SAT starts here

        # Define binary variables for c, l, r, alpha, beta, gamma, delta, etc.
        c = [[model.addVar(vtype=GRB.BINARY, name=f'c_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(self.max_adder + 2)]
        l = [[model.addVar(vtype=GRB.BINARY, name=f'l_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        r = [[model.addVar(vtype=GRB.BINARY, name=f'r_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        alpha = [[model.addVar(vtype=GRB.BINARY, name=f'alpha_{i}_{a}') for a in range(i)] for i in range(1, self.max_adder + 1)]
        beta = [[model.addVar(vtype=GRB.BINARY, name=f'beta_{i}_{a}') for a in range(i)] for i in range(1, self.max_adder + 1)]

        # c0,w is always 0 except at index fracW
        for w in range(self.fracW + 1, self.adder_wordlength):
            model.addConstr(c[0][w] == 0)
        for w in range(self.fracW):
            model.addConstr(c[0][w] == 0)
        model.addConstr(c[0][self.fracW] == 1)

        # Bound ci,0 to be an odd number
        for i in range(1, self.max_adder + 1):
            model.addConstr(c[i][0] == 1)

        # Last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            model.addConstr(c[self.max_adder + 1][w] == 0)

        # Input multiplexer constraints
        for i in range(1, self.max_adder + 1):
            alpha_sum = gp.LinExpr()
            beta_sum = gp.LinExpr()
            for a in range(i):
                for word in range(self.adder_wordlength):
                    # Equivalent to clause1_1 and clause1_2
                    model.addConstr((1 - alpha[i-1][a]) + (1 - c[a][word]) + l[i-1][word] >= 1)
                    model.addConstr((1 - alpha[i-1][a]) + c[a][word] + (1 - l[i-1][word]) >= 1)
                    
                    # Equivalent to clause2_1 and clause2_2
                    model.addConstr((1 - beta[i-1][a]) + (1 - c[a][word]) + r[i-1][word] >= 1)
                    model.addConstr((1 - beta[i-1][a]) + c[a][word] + (1 - r[i-1][word]) >= 1)
                
                alpha_sum += alpha[i-1][a]
                beta_sum += beta[i-1][a]

            # AtMost and AtLeast constraints for alpha and beta sums
            model.addConstr(alpha_sum == 1)
            model.addConstr(beta_sum == 1)

        # Left Shifter constraints
        gamma = [[model.addVar(vtype=GRB.BINARY, name=f'gamma_{i}_{k}') for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]
        s = [[model.addVar(vtype=GRB.BINARY, name=f's_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            gamma_sum = gp.LinExpr()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    # Equivalent to clause3_1 and clause3_2
                    model.addConstr((1 - gamma[i-1][k]) + (1 - l[i-1][j]) + s[i-1][j+k] >= 1)
                    model.addConstr((1 - gamma[i-1][k]) + l[i-1][j] + (1 - s[i-1][j+k]) >= 1)

                gamma_sum += gamma[i-1][k]
            
            model.addConstr(gamma_sum == 1)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    # Equivalent to clause4, clause5, and clause6
                    model.addConstr((1 - gamma[i-1][kf]) + (1 - s[i-1][b]) >= 1)
                    model.addConstr((1 - gamma[i-1][kf]) + (1 - l[i-1][self.adder_wordlength - 1]) + l[i-1][self.adder_wordlength - 2 - b] >= 1)
                    model.addConstr((1 - gamma[i-1][kf]) + l[i-1][self.adder_wordlength - 1] + (1 - l[i-1][self.adder_wordlength - 2 - b]) >= 1)

            # Equivalent to clause7_1 and clause7_2
            model.addConstr((1 - l[i-1][self.adder_wordlength - 1]) + s[i-1][self.adder_wordlength - 1] >= 1)
            model.addConstr(l[i-1][self.adder_wordlength - 1] + (1 - s[i-1][self.adder_wordlength - 1]) >= 1)

        # Delta selector constraints
        delta = [model.addVar(vtype=GRB.BINARY, name=f'delta_{i}') for i in range(1, self.max_adder + 1)]
        u = [[model.addVar(vtype=GRB.BINARY, name=f'u_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        x = [[model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                # Equivalent to clause8_1 and clause8_2
                model.addConstr((1 - delta[i-1]) + (1 - s[i-1][word]) + x[i-1][word] >= 1)
                model.addConstr((1 - delta[i-1]) + s[i-1][word] + (1 - x[i-1][word]) >= 1)

                # Equivalent to clause9_1 and clause9_2
                model.addConstr((1 - delta[i-1]) + (1 - r[i-1][word]) + u[i-1][word] >= 1)
                model.addConstr((1 - delta[i-1]) + r[i-1][word] + (1 - u[i-1][word]) >= 1)

                # Equivalent to clause10_1 and clause10_2
                model.addConstr(delta[i-1] + (1 - s[i-1][word]) + u[i-1][word] >= 1)
                model.addConstr(delta[i-1] + s[i-1][word] + (1 - u[i-1][word]) >= 1)

                # Equivalent to clause11_1 and clause11_2
                model.addConstr(delta[i-1] + (1 - r[i-1][word]) + x[i-1][word] >= 1)
                model.addConstr(delta[i-1] + r[i-1][word] + (1 - x[i-1][word]) >= 1)

        # XOR constraints
        epsilon = [model.addVar(vtype=GRB.BINARY, name=f'epsilon_{i}') for i in range(1, self.max_adder + 1)]
        y = [[model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                # Equivalent to clause12, clause13, clause14, clause15
                model.addConstr((1 - u[i-1][word]) + (1 - epsilon[i-1]) + y[i-1][word] >= 1)
                model.addConstr((1 - u[i-1][word]) + epsilon[i-1] + (1 - y[i-1][word]) >= 1)
                model.addConstr(u[i-1][word] + (1 - epsilon[i-1]) + (1 - y[i-1][word]) >= 1)
                model.addConstr(u[i-1][word] + epsilon[i-1] + y[i-1][word] >= 1)

        # Ripple carry constraints
        z = [[model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        cout = [[model.addVar(vtype=GRB.BINARY, name=f'cout_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            model.addConstr((1 - x[i-1][0]) + (1 - y[i-1][0]) + (1 - epsilon[i-1]) + z[i-1][0] >= 1)
            model.addConstr((1 - x[i-1][0]) + (1 - y[i-1][0]) + epsilon[i-1] + (1 - z[i-1][0]) >= 1)
            model.addConstr((1 - x[i-1][0]) + y[i-1][0] + (1 - epsilon[i-1]) + (1 - z[i-1][0]) >= 1)
            model.addConstr(x[i-1][0] + (1 - y[i-1][0]) + (1 - epsilon[i-1]) + z[i-1][0] >= 1)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            model.addConstr((1 - x[i-1][0]) + (1 - y[i-1][0]) + cout[i-1][0] >= 1)
            model.addConstr(x[i-1][0] + y[i-1][0] + (1 - cout[i-1][0]) >= 1)
            model.addConstr((1 - x[i-1][0]) + (1 - epsilon[i-1]) + cout[i-1][0] >= 1)
            model.addConstr(x[i-1][0] + epsilon[i-1] + (1 - cout[i-1][0]) >= 1)

            for kf in range(1, self.adder_wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                model.addConstr((1 - x[i-1][kf]) + (1 - y[i-1][kf]) + (1 - cout[i-1][kf-1]) + z[i-1][kf] >= 1)
                model.addConstr((1 - x[i-1][kf]) + y[i-1][kf] + (1 - cout[i-1][kf-1]) + (1 - z[i-1][kf]) >= 1)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                model.addConstr((1 - x[i-1][kf]) + (1 - y[i-1][kf]) + cout[i-1][kf] >= 1)
                model.addConstr(x[i-1][kf] + y[i-1][kf] + (1 - cout[i-1][kf]) >= 1)

            model.addConstr(epsilon[i-1] + x[i-1][self.adder_wordlength-1] + u[i-1][self.adder_wordlength-1] + (1 - z[i-1][self.adder_wordlength-1]) >= 1)

        # Right shift constraints
        zeta = [[model.addVar(vtype=GRB.BINARY, name=f'zeta_{i}_{k}') for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            zeta_sum = gp.LinExpr()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    # Equivalent to clause48_1 and clause48_2
                    model.addConstr((1 - zeta[i-1][k]) + (1 - z[i-1][j+k]) + c[i][j] >= 1)
                    model.addConstr((1 - zeta[i-1][k]) + z[i-1][j+k] + (1 - c[i][j]) >= 1)

                zeta_sum += zeta[i-1][k]
            
            model.addConstr(zeta_sum == 1)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    # Equivalent to clause49_1, clause49_2, clause50
                    model.addConstr((1 - zeta[i-1][kf]) + (1 - z[i-1][self.adder_wordlength - 1]) + c[i][self.adder_wordlength - 2 - b] >= 1)
                    model.addConstr((1 - zeta[i-1][kf]) + z[i-1][self.adder_wordlength - 1] + (1 - c[i][self.adder_wordlength - 2 - b]) >= 1)
                    model.addConstr((1 - zeta[i-1][kf]) + (1 - z[i-1][b]) >= 1)

            # Equivalent to clause51_1 and clause51_2
            model.addConstr((1 - z[i-1][self.adder_wordlength - 1]) + c[i][self.adder_wordlength - 1] >= 1)
            model.addConstr(z[i-1][self.adder_wordlength - 1] + (1 - c[i][self.adder_wordlength - 1]) >= 1)

        # Set connected coefficient
        connected_coefficient = half_order + 1 - self.avail_dsp

        # Solver connection
        theta = [[model.addVar(vtype=GRB.BINARY, name=f'theta_{i}_{m}') for m in range(half_order + 1)] for i in range(self.max_adder + 2)]
        iota = [model.addVar(vtype=GRB.BINARY, name=f'iota_{m}') for m in range(half_order + 1)]
        t = [[model.addVar(vtype=GRB.BINARY, name=f't_{m}_{w}') for w in range(self.adder_wordlength)] for m in range(half_order + 1)]

        iota_sum = gp.LinExpr()
        for i in range(self.max_adder + 2):
            theta_or = gp.LinExpr()
            for m in range(half_order + 1):
                for word in range(self.adder_wordlength):
                    # Equivalent to clause52_1 and clause52_2
                    model.addConstr((1 - theta[i][m]) + (1 - iota[m]) + (1 - c[i][word]) + t[m][word] >= 1)
                    model.addConstr((1 - theta[i][m]) + (1 - iota[m]) + c[i][word] + (1 - t[m][word]) >= 1)
                theta_or += theta[i][m]
            model.addConstr(theta_or >= 1)

        for m in range(half_order + 1):
            iota_sum += iota[m]

        model.addConstr(iota_sum == connected_coefficient)

        # Left shifter in result module
        h_ext = [[model.addVar(vtype=GRB.BINARY, name=f'h_ext_{m}_{w}') for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
        phi = [[model.addVar(vtype=GRB.BINARY, name=f'phi_{m}_{k}') for k in range(self.adder_wordlength - 1)] for m in range(half_order + 1)]

        for m in range(half_order + 1):
            phi_sum = gp.LinExpr()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    # Equivalent to clause53_1 and clause53_2
                    model.addConstr((1 - phi[m][k]) + (1 - t[m][j]) + h_ext[m][j+k] >= 1)
                    model.addConstr((1 - phi[m][k]) + t[m][j] + (1 - h_ext[m][j+k]) >= 1)
                
                phi_sum += phi[m][k]

            model.addConstr(phi_sum == 1)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    # Equivalent to clause54, clause55, clause56
                    model.addConstr((1 - phi[m][kf]) + (1 - h_ext[m][b]) >= 1)
                    model.addConstr((1 - phi[m][kf]) + (1 - t[m][self.adder_wordlength - 1]) + t[m][self.adder_wordlength - 2 - b] >= 1)
                    model.addConstr((1 - phi[m][kf]) + t[m][self.adder_wordlength - 1] + (1 - t[m][self.adder_wordlength - 2 - b]) >= 1)

            # Equivalent to clause57_1 and clause57_2
            model.addConstr((1 - t[m][self.adder_wordlength - 1]) + h_ext[m][self.adder_wordlength - 1] >= 1)
            model.addConstr(t[m][self.adder_wordlength - 1] + (1 - h_ext[m][self.adder_wordlength - 1]) >= 1)

            for word in range(self.adder_wordlength):
                if word <= self.wordlength - 1:
                    # Equivalent to clause58 and clause59
                    model.addConstr(h[m][word] + (1 - h_ext[m][word]) >= 1)
                    model.addConstr((1 - h[m][word]) + h_ext[m][word] >= 1)
                else:
                    model.addConstr(h[m][self.wordlength - 1] + (1 - h_ext[m][word]) >= 1)
                    model.addConstr((1 - h[m][self.wordlength - 1]) + h_ext[m][word] >= 1)

        if self.adder_depth > 0:
            # Binary variables for psi_alpha and psi_beta
            psi_alpha = [[model.addVar(vtype=GRB.BINARY, name=f'psi_alpha_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.max_adder+1)]
            psi_beta = [[model.addVar(vtype=GRB.BINARY, name=f'psi_beta_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.max_adder+1)]

            psi_alpha_sum = []
            psi_beta_sum = []

            for i in range(1, self.max_adder+1):
                # Clause 60: Not(psi_alpha[i-1][0]) or alpha[i-1][0]
                model.addConstr(psi_alpha[i-1][0] + (1 - alpha[i-1][0]) >= 1)
                
                # Clause 61: Not(psi_beta[i-1][0]) or beta[i-1][0]
                model.addConstr(psi_beta[i-1][0] + (1 - beta[i-1][0]) >= 1)
                
                for d in range(self.adder_depth):
                    psi_alpha_sum.append(psi_alpha[i-1][d])
                    psi_beta_sum.append(psi_beta[i-1][d])

                # AtMost and AtLeast for psi_alpha_sum and psi_beta_sum
                model.addConstr(sum(psi_alpha_sum) <= 1)
                model.addConstr(sum(psi_alpha_sum) >= 1)
                model.addConstr(sum(psi_beta_sum) <= 1)
                model.addConstr(sum(psi_beta_sum) >= 1)

                if d == 1:
                    continue

                for d in range(1, self.adder_depth):
                    for a in range(i-1):
                        # Clause 63: Not(psi_alpha[i-1][d]) or alpha[i-1][a]
                        model.addConstr(psi_alpha[i-1][d] + (1 - alpha[i-1][a]) >= 1)

                        # Clause 64: Not(psi_alpha[i-1][d]) or psi_alpha[i-1][d-1]
                        model.addConstr(psi_alpha[i-1][d] + (1 - psi_alpha[i-1][d-1]) >= 1)

                        # Clause 65: Not(psi_beta[i-1][d]) or beta[i-1][a]
                        model.addConstr(psi_beta[i-1][d] + (1 - beta[i-1][a]) >= 1)

                        # Clause 66: Not(psi_beta[i-1][d]) or psi_beta[i-1][d-1]
                        model.addConstr(psi_beta[i-1][d] + (1 - psi_beta[i-1][d-1]) >= 1)

        
        start_time=time.time()
        model.optimize()


        print("solver runing")


        # print(filter_coeffs)
        # print(filter_literals)


        if model.Status == GRB.TIME_LIMIT:
            end_time = time.time()

            satifiability = 'Timeout'

        elif model.status == GRB.OPTIMAL:
            end_time = time.time()

            satifiability = 'sat'

            print("solver sat")

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = h[m][w].X
                    print(bool_value)
                    print(f"h{m}_{w} = ",bool_value)
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_res)

           # Store gain coefficient
            self.result_model[f"gain"] = gain.X

            # Store h coefficients
            for m in range(half_order + 1):
                self.result_model[f"h[{m}]"] = self.h_res[m]

            # Store alpha and beta selectors
            for i in range(len(alpha)):
                for a in range(len(alpha[i])):
                    self.result_model[f'alpha[{i+1}][{a}]'] = alpha[i][a].X
                for a in range(len(beta[i])):
                    self.result_model[f'beta[{i+1}][{a}]'] = beta[i][a].X

            # Store gamma (left shift selectors)
            for i in range(len(gamma)):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'gamma[{i+1}][{k}]'] = gamma[i][k].X

            # Store delta selectors and u/x arrays
            for i in range(len(delta)):
                self.result_model[f'delta[{i+1}]'] = delta[i].X

            # Store epsilon selectors and y array (XOR results)
            for i in range(len(epsilon)):
                self.result_model[f'epsilon[{i+1}]'] = epsilon[i].X

            # Store zeta (right shift selectors)
            for i in range(len(zeta)):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'zeta[{i+1}][{k}]'] = zeta[i][k].X

            # Store theta array
            for i in range(len(theta)):
                for m in range(half_order + 1):
                    self.result_model[f'theta[{i+1}][{m}]'] = theta[i][m].X

            # Store iota array
            for m in range(len(iota)):
                self.result_model[f'iota[{m}]'] = iota[m].X


            
            
            
            self.gain_res=gain.x
            print("gain Coeffs: ", self.gain_res)
                      
        else:
            satifiability = 'unsat'
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        model.dispose()  # Dispose of the model
        del model

        return duration , satifiability


        

                    

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
        self.freq_upper_lin = ((freq_upper_lin_array/((10**self.coef_accuracy)*(2**(self.fracW)))) * self.gain_res).tolist()
        self.freq_lower_lin = ((freq_lower_lin_array/((10**self.coef_accuracy)*(2**(self.fracW)))) * self.gain_res).tolist()


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
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += self.h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            computed_frequency_response.append(np.abs(term_sum_exprs))
        
        # Normalize frequencies to range from 0 to 1 for plotting purposes

        # Plot the computed frequency response
        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        self.ax2.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)


        if self.app:
            self.app.canvas.draw()



    

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 30
    accuracy = 2
    adder_count = 3
    wordlength = 10

    space = int(accuracy*order_upper)
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.5*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 3
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -5
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterGurobi(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()

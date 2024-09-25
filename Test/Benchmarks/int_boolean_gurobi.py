import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import math
import random


class SolverFunc():
    def __init__(self,input_data):
        self.filter_type = None
        self.order_upperbound = None

        self.original_xdata = None
        self.original_upperbound_lin = None
        self.original_lowerbound_lin = None

        self.cutoffs_x = None
        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.solver_accuracy_multiplier = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
        
        self.overflow_count = 0

    
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
        





class FIRFilter:
    def __init__(self, 
                 filter_type, 
                 order, 
                 freqx_axis, 
                 upperbound_lin, 
                 lowerbound_lin, 
                 ignore_lowerbound, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 coef_accuracy,
                 intW,
                 app = None
                 ):
       
        
        self.filter_type = filter_type
        self.order = order
        self.freqx_axis = freqx_axis

        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.max_adder = adder_count

        self.upperbound_lin=upperbound_lin
        self.lowerbound_lin=lowerbound_lin

        self.coef_accuracy = coef_accuracy
        self.intW = intW
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound= gain_upperbound
        self.gain_lowerbound= gain_lowerbound

        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.adder_wordlength = self.wordlength + adder_wordlength_ext
        self.result_model = {}

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order,
        }

        return input_data_sf
    
    def run_barebone(self, thread, minmax_option = None, h_zero_count = None):
        

        self.h_res = []
        self.gain_res = 0
        target_result = {}
        self.order_current = int(self.order)
        half_order = (self.order_current // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2) - 1
        
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

        # print(f"upperbound_lin: {self.upperbound_lin}")
        # print(f"lowerbound_lin: {self.lowerbound_lin}")


         # linearize the bounds
        internal_upperbound_lin = [math.ceil((f)*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.floor((f)*(10**self.coef_accuracy)*(2**(self.fracW))) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound*(10**self.coef_accuracy)*(2**self.fracW)


        # print("Running Gurobi with the following parameters:")
        # print(f"thread: {thread}")
        # print(f"minmax_option: {minmax_option}")
        # print(f"h_zero_count: {h_zero_count}")
        # print(f"filter_type: {self.filter_type}")
        # print(f"order_current: {self.order}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")
        
        model = gp.Model()
        model.setParam('Threads', thread)
        model.setParam('OutputFlag', 0)
        model.setObjective(0, GRB.MINIMIZE)



        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")



        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
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
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain*internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain*-internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain*internal_lowerbound_lin[omega])
        

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
        for m in range(half_order + 1):
            theta_or = gp.LinExpr()
            for i in range(self.max_adder + 2):
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
        print("Gurobi: Barebone running")
        model.optimize()

        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'
            end_time = time.time()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = h[m][w].X
                    
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            # print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res=gain.x
            #print("gain Coeffs: ", self.gain_res)
            
                
            

                      
        else:
            print("Gurobi: Unsatisfiable")
            end_time = time.time()
            
            
        model.terminate()  # Dispose of the model
        del model

        duration = end_time - start_time

        return satisfiability, self.h_res ,duration
        

    def run_barebone_real(self,thread, minmax_option ,h_zero_count = None, h_target = None):
        self.h_res = []
        self.gain_res = []

        self.order_current = int(self.order)
        half_order = (self.order_current // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_current // 2) - 1
        
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

         # linearize the bounds
        internal_upperbound_lin = [math.floor((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.ceil((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound*(10**self.coef_accuracy)

        # print("filter order:", self.order_current)
        # print(" filter_type:", self.filter_type)
        # print("freqx_axis:", self.freqx_axis)
        # print("ignore lower than:", internal_ignore_lowerbound)
        
        # print(f"lower {internal_lowerbound_lin}")
        # print(f"upper {internal_upperbound_lin}")
        # print(f"coef_accuracy {self.coef_accuracy}")
        # print(f"gain_upperbound {self.gain_upperbound}")
        # print(f"gain_lowerbound {self.gain_lowerbound}")

        
        model = gp.Model(f"presolve_model_{minmax_option}")
        model.setParam('Threads', thread)
        model.setParam('OutputFlag', 0)
        model.setObjective(0, GRB.MAXIMIZE)

        


        h_upperbound = ((2**(self.intW-1))-1)+(1-2**-self.fracW)
        h_lowerbound = -2**(self.intW-1)

        h_real = [model.addVar(vtype=GRB.CONTINUOUS,lb=h_lowerbound, ub=h_upperbound, name=f'h_real_{a}') for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                cm_word_prod= int(cm*(10** self.coef_accuracy))
                h_sum_temp += h_real[m]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain*internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain*-internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain*internal_lowerbound_lin[omega])
        
        print(f"Gurobi barebone_real: {minmax_option}")
        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]

        for m in range(half_order + 1):
            h_sum_temp = 0
            for w in range(self.wordlength):
                weight = 0
                if w==self.wordlength-1:
                    weight = -2**(w-self.fracW)  
                elif w < self.fracW:                   
                    weight = 2**(-1*(self.fracW-w)) 
                else:
                    weight = 2**(w-self.fracW)

                h_sum_temp+= h[m][w]*weight

            model.addConstr(h_sum_temp == h_real[m])

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
        for m in range(half_order + 1):
            theta_or = gp.LinExpr()
            for i in range(self.max_adder + 2):
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



        print("Gurobi: MinMax running")
        start_time=time.time()
        model.optimize()
        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'
            end_time = time.time()
            h_res_bool = []
            for m in range(half_order + 1):
                h_value = h_real[m].X
                self.h_res.append(h_value)
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = h[m][w].X
                    
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                h_res_bool.append(fir_coef)
            # print("FIR Coeffs calculated: ",self.h_res)
            # print("FIR Coeffs bool from real calculated: ",h_res_bool)

            self.gain_res = gain.x
            # print("gain Coeffs: ", self.gain_res)

    
                      
        else:
            print("Gurobi: Unsatisfiable")
            end_time = time.time()
            

        model.dispose()  # Dispose of the model
        del model

        duration = end_time - start_time
        return satisfiability, self.h_res ,duration










def generate_freq_bounds(space, multiplier_to_test ,order_current):
   #random bounds generator
    random.seed(it)
    lower_cutoff = random.choice([0.4, 0.5])
    upper_cutoff = random.choice([ 0.6, 0.65])
    

    lower_half_point = int(lower_cutoff * space)
    upper_half_point = int(upper_cutoff * space)
   
    
    end_point = space
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)
    passband_upperbound = random.choice([ 0 ,2 ])
    passband_lowerbound = random.choice([-2, -3 ])
    stopband_upperbound = random.choice([-20,-10])

    stopband_lowerbound = -1000
    
    freq_upper[0:lower_half_point] = passband_upperbound
    freq_lower[0:lower_half_point] = passband_lowerbound

    freq_upper[lower_half_point:upper_half_point] = passband_upperbound
    freq_lower[lower_half_point:upper_half_point] = stopband_lowerbound

    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound

    space_to_test = space * multiplier_to_test
    original_end_point = space_to_test
    original_freqx_axis = np.linspace(0, 1, space_to_test)
    original_freq_upper = np.full(space_to_test, np.nan)
    original_freq_lower = np.full(space_to_test, np.nan)

    
    original_lower_half_point = np.abs(original_freqx_axis - ((lower_half_point-1)/space)).argmin()
    original_upper_half_point = np.abs(original_freqx_axis - ((upper_half_point+1)/space)).argmin()
   
    original_freq_upper[0:original_lower_half_point] = passband_upperbound
    original_freq_lower[0:original_lower_half_point] = passband_lowerbound

    original_freq_upper[original_lower_half_point:original_upper_half_point] = passband_upperbound
    original_freq_lower[original_lower_half_point:original_upper_half_point] = stopband_lowerbound

    original_freq_upper[original_upper_half_point:original_end_point] = stopband_upperbound
    original_freq_lower[original_upper_half_point:original_end_point] = stopband_lowerbound



     #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40


    
    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]

    original_upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_upper]
    original_lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in original_freq_lower]


    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    return freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin

global it
it = 1

if __name__ == "__main__":
    
    filter_type = 0
    wordlength = 13
    gain_upperbound = 2
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 4
    

    with open("int_vs_boolean.txt", "w") as file:
        file.write("satisfiability; duration;satisfiability_bool; duration_bool;int_faster_flag;bool_faster_flag\n")

    # Test inputs from 1 to 20 accuracy multiplier
    for i in range(1,3):
        for order_current in range(14,50,2):
            accuracy = i
            space = order_current * accuracy
            freqx_axis,upperbound_lin,lowerbound_lin,ignore_lowerbound_lin,original_freqx_axis,original_upperbound_lin,original_lowerbound_lin = generate_freq_bounds(space,100,order_current)
            adder_count = order_current // 2
            #higher resolution list to test the result
            # print(np.array(freqx_axis).tolist())
            # print(np.array(upperbound_lin).tolist())
            # print(np.array(lowerbound_lin).tolist())
            it += 1

            # Create FIRFilter instance
            fir_filter = FIRFilter(
                        filter_type, 
                        order_current, 
                        freqx_axis, 
                        upperbound_lin, 
                        lowerbound_lin, 
                        ignore_lowerbound_lin, 
                        adder_count, 
                        wordlength, 
                        adder_depth,
                        avail_dsp,
                        adder_wordlength_ext,
                        gain_upperbound,
                        gain_lowerbound,
                        coef_accuracy,
                        intW,
                        )
            
            duration_bool = 0 
            satisfiability_bool = 0
            
            # Run solver and plot result
            satisfiability, h_res ,duration = fir_filter.run_barebone_real(0,'None')
            print("FIR real Coeffs calculated: ",h_res)
            satisfiability_bool,h_res_bool,duration_bool = fir_filter.run_barebone(0,'None')
            print("FIR bool Coeffs calculated: ",h_res_bool)

            
            int_faster_flag = 0
            bool_faster_flag = 0
            if duration_bool < duration:
                bool_faster_flag = 1
            else: int_faster_flag = 1

            


            with open("int_vs_boolean.txt", "a") as file:
                file.write(f"{satisfiability}; {duration};{satisfiability_bool};{duration_bool};{int_faster_flag};{bool_faster_flag}\n")
                
            print("Test ", i, " is completed")

    print("Benchmark completed and results saved to accuracy_multipier_test.txt")

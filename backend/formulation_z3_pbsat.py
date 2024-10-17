import numpy as np
from z3 import *
import matplotlib.pyplot as plt
import time
import math

try:
    from .solver_func import SolverFunc
except:
    from solver_func import SolverFunc


class FIRFilterZ3:
    def __init__(self, 
                 filter_type, 
                 half_order, 
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
                 gain_wordlength,
                 gain_intW
                 ):
        
        self.filter_type = filter_type
        self.half_order = half_order
        self.freqx_axis = freqx_axis
        self.upperbound_lin = upperbound_lin
        self.lowerbound_lin = lowerbound_lin
        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.max_adder = adder_count
        self.adder_wordlength = self.wordlength + adder_wordlength_ext

        self.coef_accuracy = coef_accuracy
        self.intW = intW
        self.fracW = self.wordlength - self.intW

        
        self.gain_wordlength=gain_wordlength #bits wordlength for gain
        self.gain_intW = gain_intW
        self.gain_fracW =  self.gain_wordlength - self.gain_intW

        self.gain_upperbound= gain_upperbound
        self.gain_lowerbound= gain_lowerbound
        self.gain_bound_accuracy = 2 #2 floating points


        self.ignore_lowerbound_lin = ignore_lowerbound_lin

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.result_model = {}

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        }

        return input_data_sf


    def run_barebone(self , thread , z3_option = None, h_zero_count = None):
        if h_zero_count == None:
            h_zero_count = 0
        self.h_res = []
        self.gain_res = 0

        ctx = z3.Context()

        half_order = self.half_order - 1 #-1 is because i am lazy to change the code
        
        sf = SolverFunc(self.get_solver_func_dict())

        # linearize the bounds
        internal_upperbound_lin = [math.floor(f*(10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW))) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.ceil(f*(10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW))) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound_lin*(10**self.coef_accuracy)*(2**(self.fracW-self.gain_fracW))


        solver = Solver(ctx=ctx)
        solver.set("random_seed",0)
        solver.set("smt.threads", thread)  

        h = [[Bool(f'h_{a}_{w}', ctx=ctx) for w in range(self.wordlength)] for a in range(half_order+1)]
        gain= [Bool(f'gain_{g}', ctx=ctx) for g in range(self.gain_wordlength)]

        gain_coeffs = []
        gain_literals=[]


        #bounds the gain
        self.gain_upperbound_int = math.floor(self.gain_upperbound*2**self.gain_fracW*(10**self.gain_bound_accuracy))
        self.gain_lowerbound_int = math.ceil(self.gain_lowerbound*2**self.gain_fracW*(10**self.gain_bound_accuracy))

        # print(self.gain_upperbound_int)
        # print(self.gain_lowerbound_int)

        # print("Running Gurobi with the following parameters:")
        # print(f"h_zero_count: {h_zero_count}")
        # print(f"filter_type: {self.filter_type}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")

        

        for g in range(self.gain_wordlength):
            gain_coeffs.append((2**g)*(10**self.gain_bound_accuracy))
            gain_literals.append(gain[g])

        pb_gain_pairs = [(gain_literals[i],gain_coeffs[i]) for i in range(len(gain_literals))]
            
        solver.add(PbLe(pb_gain_pairs, self.gain_upperbound_int))
        solver.add(PbGe(pb_gain_pairs, self.gain_lowerbound_int))
            

            
        filter_literals = []
        filter_coeffs = []
        gain_upperbound_lin_prod_coeffs = []
        gain_lowerbound_lin_prod_coeffs = []

        filter_upper_pb_pairs = []
        filter_lower_pb_pairs = []

        gain_upper_literals = []
        gain_lower_literals = []


        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            #clearing each list like this make the programm run faster, instead of decalring new one each time
            gain_literals.clear()
            filter_literals.clear()
            filter_coeffs.clear()

            gain_upperbound_lin_prod_coeffs.clear()
            gain_lowerbound_lin_prod_coeffs.clear()

            filter_upper_pb_pairs.clear()
            filter_lower_pb_pairs.clear()
            
            gain_upper_literals.clear()
            gain_lower_literals.clear()

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= int(cm*(10** self.coef_accuracy)*(-1*(2**w)))
                    else: cm_word_prod= int(cm*(10** self.coef_accuracy)*(2**w))
                    filter_literals_temp, filter_coeffs_temp = sf.overflow_handler(cm_word_prod,h[m][w])
                    filter_coeffs.extend(filter_coeffs_temp)
                    filter_literals.extend(filter_literals_temp)

            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2**g) * internal_upperbound_lin[omega])
                gain_upper_literals_temp, gain_upperbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_upper_prod,gain[g])
                gain_upperbound_lin_prod_coeffs.extend(gain_upperbound_lin_prod_coeffs_temp)
                gain_upper_literals.extend(gain_upper_literals_temp)

                if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                    gain_lower_prod=int((2**g) * internal_upperbound_lin[omega])
                    gain_lower_literals_temp, gain_lowerbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod,gain[g])
                    gain_lowerbound_lin_prod_coeffs.extend(gain_lowerbound_lin_prod_coeffs_temp)
                    gain_lower_literals.extend(gain_lower_literals_temp)
                    # print("ignored ",internal_lowerbound_lin[omega], " in frequency = ", self.freqx_axis[omega])
                else:
                    gain_lower_prod=int(-1 *(2**g) * internal_lowerbound_lin[omega])
                    gain_lower_literals_temp, gain_lowerbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod,gain[g])
                    gain_lowerbound_lin_prod_coeffs.extend(gain_lowerbound_lin_prod_coeffs_temp)
                    gain_lower_literals.extend(gain_lower_literals_temp)

            filter_upper_pb_coeffs=filter_coeffs+gain_upperbound_lin_prod_coeffs
            filter_upper_pb_literals=filter_literals+gain_upper_literals


            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literals):
                raise("sumtin wong with upper filter pb")


            #z3 only take pairs
            filter_upper_pb_pairs = [(filter_upper_pb_literals[i],filter_upper_pb_coeffs[i],) for i in range(len(filter_upper_pb_literals))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            
            filter_lower_pb_coeffs=filter_coeffs+gain_lowerbound_lin_prod_coeffs
            filter_lower_pb_literals=filter_literals+gain_lower_literals


            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literals):
                raise("sumtin wong with upper filter pb")
            
            
            filter_lower_pb_pairs = [(filter_lower_pb_literals[i],filter_lower_pb_coeffs[i]) for i in range(len(filter_lower_pb_literals))]
            
            #z3 only take pairs
            solver.add(PbGe(filter_lower_pb_pairs, 0))

            #end omega loop

        if z3_option == 'try_h_zero_count':
            print("z3_presolve: try_h_zero_count")
            if h_zero_count == None:
                raise TypeError("z3 barebone: h_zero_count cant be empty when try_h_zero_count is chosen")
            h_zero_sum = []
            h_zero = [Bool(f'h_zero_{m}', ctx=ctx) for m in range(half_order+1)]
            for m in range(half_order+1):
                for word in range(self.wordlength):
                    solver.add(Or(Not(h_zero[m]),Not(h[m][word]))) 
                h_zero_sum.append(h_zero[m])
            solver.add(AtLeast(*h_zero_sum,h_zero_count))


        print(f"Z3: Barebone running with thread {thread}")
        satisfiability = 'unsat'

        if solver.check() == sat:

            satisfiability = 'sat'

            model = solver.model()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = model.eval(h[m][w], model_completion=True)

                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_res)
            
            gain_coef=0
            for g in range(self.gain_wordlength):
                # Evaluate the boolean value from the model
                bool_value = model.eval(gain[g], model_completion=True)
                # Convert boolean to integer (0 or 1) and calculate the term

                if g < self.gain_fracW:                   
                    gain_coef += 2**-(self.gain_fracW-g) * (1 if bool_value else 0)
                else: 
                    gain_coef += 2**(g-self.gain_fracW) * (1 if bool_value else 0)

            self.gain_res=gain_coef


        else:
            print("Unsatisfiable")

        print(f"Z3: Barebone done solving with thread {thread}")
        print(satisfiability)
        print(self.h_res)
        print(self.gain_res)

        return satisfiability, self.h_res , self.gain_res


    def runsolver(self, thread, adderm, h_zero_count = 0):
        self.result_model = {}
        self.max_adder = adderm
        self.h_res = []
        self.gain_res = 0

        ctx = z3.Context()


        half_order = self.half_order - 1 #-1 is because i am lazy to change the code

        print("solver called")
        print("ignore lower than:", self.ignore_lowerbound_lin)
        # linearize the bounds
        internal_upperbound_lin = [
            math.ceil(f * (10 ** self.coef_accuracy) * (2 ** (self.fracW - self.gain_fracW))) if not np.isnan(f) else np.nan
            for f in self.upperbound_lin
        ]
        internal_lowerbound_lin = [
            math.floor(f * (10 ** self.coef_accuracy) * (2 ** (self.fracW - self.gain_fracW))) if not np.isnan(f) else np.nan
            for f in self.lowerbound_lin
        ]
        internal_ignore_lowerbound = self.ignore_lowerbound_lin * (10 ** self.coef_accuracy) * (2 ** (self.fracW - self.gain_fracW))





        
        solver = Solver(ctx=ctx)
        solver.set("random_seed",0)
        solver.set("smt.threads", thread)  

        h = [[Bool(f'h_{a}_{w}', ctx=ctx) for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = [Bool(f'gain_{g}', ctx=ctx) for g in range(self.gain_wordlength)]

        if h_zero_count > 0:
            print("z3_running: try_h_zero_count")
            h_zero_sum = []
            h_zero = [Bool(f'h_zero_{m}', ctx=ctx) for m in range(half_order+1)]
            for m in range(half_order+1):
                for word in range(self.wordlength):
                    solver.add(Or(Not(h_zero[m]),Not(h[m][word]))) 
                h_zero_sum.append(h_zero[m])
            solver.add(AtLeast(*h_zero_sum,h_zero_count))

        sf = SolverFunc(self.get_solver_func_dict())

        gain_coeffs = []
        gain_literals = []

        # bounds the gain
        self.gain_upperbound_int = int(self.gain_upperbound * 2 ** self.gain_fracW * (10 ** self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound * 2 ** self.gain_fracW * (10 ** self.gain_bound_accuracy))

        # print(self.gain_upperbound_int)
        # print(self.gain_lowerbound_int)

        for g in range(self.gain_wordlength):
            gain_coeffs.append((2 ** g) * (10 ** self.gain_bound_accuracy))
            gain_literals.append(gain[g])

        pb_gain_pairs = [(gain_literals[i], gain_coeffs[i]) for i in range(len(gain_literals))]

        solver.add(PbLe(pb_gain_pairs, self.gain_upperbound_int))
        solver.add(PbGe(pb_gain_pairs, self.gain_lowerbound_int))

        filter_literals = []
        filter_coeffs = []
        gain_upperbound_lin_prod_coeffs = []
        gain_lowerbound_lin_prod_coeffs = []

        filter_upper_pb_pairs = []
        filter_lower_pb_pairs = []

        gain_upper_literals = []
        gain_lower_literals = []

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            # clearing each list like this make the programm run faster, instead of decalring new one each time
            gain_literals.clear()
            filter_literals.clear()
            filter_coeffs.clear()

            gain_upperbound_lin_prod_coeffs.clear()
            gain_lowerbound_lin_prod_coeffs.clear()

            filter_upper_pb_pairs.clear()
            filter_lower_pb_pairs.clear()

            gain_upper_literals.clear()
            gain_lower_literals.clear()

            for m in range(half_order + 1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w == self.wordlength - 1:
                        cm_word_prod = int(cm * (10 ** self.coef_accuracy) * (-1 * (2 ** w)))
                    else:
                        cm_word_prod = int(cm * (10 ** self.coef_accuracy) * (2 ** w))
                    filter_literals_temp, filter_coeffs_temp = sf.overflow_handler(cm_word_prod, h[m][w])
                    filter_coeffs.extend(filter_coeffs_temp)
                    filter_literals.extend(filter_literals_temp)

            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2 ** g) * internal_upperbound_lin[omega])
                gain_upper_literals_temp, gain_upperbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_upper_prod, gain[g])
                gain_upperbound_lin_prod_coeffs.extend(gain_upperbound_lin_prod_coeffs_temp)
                gain_upper_literals.extend(gain_upper_literals_temp)

                if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                    gain_lower_prod = int((2 ** g) * internal_upperbound_lin[omega])
                    gain_lower_literals_temp, gain_lowerbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod, gain[g])
                    gain_lowerbound_lin_prod_coeffs.extend(gain_lowerbound_lin_prod_coeffs_temp)
                    gain_lower_literals.extend(gain_lower_literals_temp)
                    # print("ignored ", internal_lowerbound_lin[omega], " in frequency = ", self.freqx_axis[omega])
                else:
                    gain_lower_prod = int(-1 * (2 ** g) * internal_lowerbound_lin[omega])
                    gain_lower_literals_temp, gain_lowerbound_lin_prod_coeffs_temp = sf.overflow_handler(gain_lower_prod, gain[g])
                    gain_lowerbound_lin_prod_coeffs.extend(gain_lowerbound_lin_prod_coeffs_temp)
                    gain_lower_literals.extend(gain_lower_literals_temp)

            filter_upper_pb_coeffs = filter_coeffs + gain_upperbound_lin_prod_coeffs
            filter_upper_pb_literals = filter_literals + gain_upper_literals

            # print("coeffs: ",filter_upper_pb_coeffs)
            # print("lit: ",filter_upper_pb_literals)

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literals):
                raise ("sumtin wong with upper filter pb")

            # else: print("filter upperbound length is validated")

            # z3 only take pairs
            filter_upper_pb_pairs = [(filter_upper_pb_literals[i], filter_upper_pb_coeffs[i],) for i in range(len(filter_upper_pb_literals))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            filter_lower_pb_coeffs = filter_coeffs + gain_lowerbound_lin_prod_coeffs
            filter_lower_pb_literals = filter_literals + gain_lower_literals

            # print("coeffs: ",filter_lower_pb_coeffs)
            # print("lit: ",filter_lower_pb_literals)

            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literals):
                raise ("sumtin wong with upper filter pb")

            # else: print("filter lowerbound length is validated")

            filter_lower_pb_pairs = [(filter_lower_pb_literals[i], filter_lower_pb_coeffs[i]) for i in range(len(filter_lower_pb_literals))]

            # z3 only take pairs
            solver.add(PbGe(filter_lower_pb_pairs, 0))

            # end omega loop

        # bitshift sat starts here

        # input multiplexer
        c = [[Bool(f'c_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(self.max_adder + 2)]
        l = [[Bool(f'l_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        r = [[Bool(f'r_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        alpha = [[Bool(f'alpha_{i}_{a}', ctx=ctx) for a in range(i)] for i in range(1, self.max_adder + 1)]
        beta = [[Bool(f'Beta_{i}_{a}', ctx=ctx) for a in range(i)] for i in range(1, self.max_adder + 1)]

        # c0,w is always 0 except 1
        # for w in range(self.fracW + 1, self.adder_wordlength):
        #     solver.add(Not(c[0][w]))

        # for w in range(self.fracW):
        #     solver.add(Not(c[0][w]))

        # solver.add(c[0][self.fracW])

        # c0,w is always 0 except at 1
        for w in range(1, self.adder_wordlength):
            solver.add(Not(c[0][w]))

        solver.add(c[0][0])

        # bound ci,0 to be odd number
        for i in range(1, self.max_adder + 1):
            solver.add(c[i][0])

        # last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            solver.add(Not(c[self.max_adder + 1][w]))

        # input multiplexer
        for i in range(1, self.max_adder + 1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.adder_wordlength):
                    clause1_1 = Or(Not(alpha[i - 1][a]), Not(c[a][word]), l[i - 1][word])
                    clause1_2 = Or(Not(alpha[i - 1][a]), c[a][word], Not(l[i - 1][word]))
                    solver.add(And(clause1_1, clause1_2))

                    clause2_1 = Or(Not(beta[i - 1][a]), Not(c[a][word]), r[i - 1][word])
                    clause2_2 = Or(Not(beta[i - 1][a]), c[a][word], Not(r[i - 1][word]))
                    solver.add(And(clause2_1, clause2_2))

                alpha_sum.append(alpha[i - 1][a])
                beta_sum.append(beta[i - 1][a])

            solver.add(AtMost(*alpha_sum, 1))
            solver.add(AtLeast(*alpha_sum, 1))

            solver.add(AtMost(*beta_sum, 1))
            solver.add(AtLeast(*beta_sum, 1))

        # Left Shifter
        # k is the shift selector
        gamma = [[Bool(f'gamma_{i}_{k}', ctx=ctx) for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]
        s = [[Bool(f's_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            gamma_sum = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    clause3_1 = Or(Not(gamma[i - 1][k]), Not(l[i - 1][j]), s[i - 1][j + k])
                    clause3_2 = Or(Not(gamma[i - 1][k]), l[i - 1][j], Not(s[i - 1][j + k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause3_1)
                    solver.add(clause3_2)

                gamma_sum.append(gamma[i - 1][k])

            solver.add(AtMost(*gamma_sum, 1))
            solver.add(AtLeast(*gamma_sum, 1))

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i - 1][kf]), Not(s[i - 1][b]))
                    clause5 = Or(Not(gamma[i - 1][kf]), Not(l[i - 1][self.adder_wordlength - 1]), l[i - 1][self.adder_wordlength - 2 - b])
                    clause6 = Or(Not(gamma[i - 1][kf]), l[i - 1][self.adder_wordlength - 1], Not(l[i - 1][self.adder_wordlength - 2 - b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)

            clause7_1 = Or(Not(l[i - 1][self.adder_wordlength - 1]), s[i - 1][self.adder_wordlength - 1])
            clause7_2 = Or(l[i - 1][self.adder_wordlength - 1], Not(s[i - 1][self.adder_wordlength - 1]))
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause7_1)
            solver.add(clause7_2)

        delta = [Bool(f'delta_{i}', ctx=ctx) for i in range(1, self.max_adder + 1)]
        u = [[Bool(f'u_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        x = [[Bool(f'x_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        # delta selector
        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                clause8_1 = Or(Not(delta[i - 1]), Not(s[i - 1][word]), x[i - 1][word])
                clause8_2 = Or(Not(delta[i - 1]), s[i - 1][word], Not(x[i - 1][word]))
                # solver.add(And(clause8_1, clause8_2))
                solver.add(clause8_1)
                solver.add(clause8_2)

                clause9_1 = Or(Not(delta[i - 1]), Not(r[i - 1][word]), u[i - 1][word])
                clause9_2 = Or(Not(delta[i - 1]), r[i - 1][word], Not(u[i - 1][word]))
                # solver.add(And(clause9_1, clause9_2))
                solver.add(clause9_1)
                solver.add(clause9_2)

                clause10_1 = Or(delta[i - 1], Not(s[i - 1][word]), u[i - 1][word])
                clause10_2 = Or(delta[i - 1], s[i - 1][word], Not(u[i - 1][word]))
                # solver.add(And(clause10_1, clause10_2))
                solver.add(clause10_1)
                solver.add(clause10_2)

                clause11_1 = Or(delta[i - 1], Not(r[i - 1][word]), x[i - 1][word])
                clause11_2 = Or(delta[i - 1], r[i - 1][word], Not(x[i - 1][word]))
                # solver.add(And(clause11_1, clause11_2))
                solver.add(clause11_1)
                solver.add(clause11_2)

        epsilon = [Bool(f'epsilon_{i}', ctx=ctx) for i in range(1, self.max_adder + 1)]
        y = [[Bool(f'y_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        # xor
        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                clause12 = Or(u[i - 1][word], epsilon[i - 1], Not(y[i - 1][word]))
                clause13 = Or(u[i - 1][word], Not(epsilon[i - 1]), y[i - 1][word])
                clause14 = Or(Not(u[i - 1][word]), epsilon[i - 1], y[i - 1][word])
                clause15 = Or(Not(u[i - 1][word]), Not(epsilon[i - 1]), Not(y[i - 1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        # ripple carry
        z = [[Bool(f'z_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]
        cout = [[Bool(f'cout_{i}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            clause16 = Or(x[i - 1][0], y[i - 1][0], epsilon[i - 1], Not(z[i - 1][0]))
            clause17 = Or(x[i - 1][0], y[i - 1][0], Not(epsilon[i - 1]), z[i - 1][0])
            clause18 = Or(x[i - 1][0], Not(y[i - 1][0]), epsilon[i - 1], z[i - 1][0])
            clause19 = Or(Not(x[i - 1][0]), y[i - 1][0], epsilon[i - 1], z[i - 1][0])
            clause20 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), Not(epsilon[i - 1]), z[i - 1][0])
            clause21 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), epsilon[i - 1], Not(z[i - 1][0]))
            clause22 = Or(Not(x[i - 1][0]), y[i - 1][0], Not(epsilon[i - 1]), Not(z[i - 1][0]))
            clause23 = Or(x[i - 1][0], Not(y[i - 1][0]), Not(epsilon[i - 1]), Not(z[i - 1][0]))

            solver.add(clause16)
            solver.add(clause17)
            solver.add(clause18)
            solver.add(clause19)
            solver.add(clause20)
            solver.add(clause21)
            solver.add(clause22)
            solver.add(clause23)

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            clause24 = Or(Not(x[i - 1][0]), Not(y[i - 1][0]), cout[i - 1][0])
            clause25 = Or(x[i - 1][0], y[i - 1][0], Not(cout[i - 1][0]))
            clause26 = Or(Not(x[i - 1][0]), Not(epsilon[i - 1]), cout[i - 1][0])
            clause27 = Or(x[i - 1][0], epsilon[i - 1], Not(cout[i - 1][0]))
            clause28 = Or(Not(y[i - 1][0]), Not(epsilon[i - 1]), cout[i - 1][0])
            clause29 = Or(y[i - 1][0], epsilon[i - 1], Not(cout[i - 1][0]))

            solver.add(clause24)
            solver.add(clause25)
            solver.add(clause26)
            solver.add(clause27)
            solver.add(clause28)
            solver.add(clause29)

            for kf in range(1, self.adder_wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                clause30 = Or(x[i - 1][kf], y[i - 1][kf], cout[i - 1][kf - 1], Not(z[i - 1][kf]))
                clause31 = Or(x[i - 1][kf], y[i - 1][kf], Not(cout[i - 1][kf - 1]), z[i - 1][kf])
                clause32 = Or(x[i - 1][kf], Not(y[i - 1][kf]), cout[i - 1][kf - 1], z[i - 1][kf])
                clause33 = Or(Not(x[i - 1][kf]), y[i - 1][kf], cout[i - 1][kf - 1], z[i - 1][kf])
                clause34 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), z[i - 1][kf])
                clause35 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), cout[i - 1][kf - 1], Not(z[i - 1][kf]))
                clause36 = Or(Not(x[i - 1][kf]), y[i - 1][kf], Not(cout[i - 1][kf - 1]), Not(z[i - 1][kf]))
                clause37 = Or(x[i - 1][kf], Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), Not(z[i - 1][kf]))

                solver.add(clause30)
                solver.add(clause31)
                solver.add(clause32)
                solver.add(clause33)
                solver.add(clause34)
                solver.add(clause35)
                solver.add(clause36)
                solver.add(clause37)

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                clause38 = Or(Not(x[i - 1][kf]), Not(y[i - 1][kf]), cout[i - 1][kf])
                clause39 = Or(x[i - 1][kf], y[i - 1][kf], Not(cout[i - 1][kf]))
                clause40 = Or(Not(x[i - 1][kf]), Not(cout[i - 1][kf - 1]), cout[i - 1][kf])
                clause41 = Or(x[i - 1][kf], cout[i - 1][kf - 1], Not(cout[i - 1][kf]))
                clause42 = Or(Not(y[i - 1][kf]), Not(cout[i - 1][kf - 1]), cout[i - 1][kf])
                clause43 = Or(y[i - 1][kf], cout[i - 1][kf - 1], Not(cout[i - 1][kf]))

                solver.add(clause38)
                solver.add(clause39)
                solver.add(clause40)
                solver.add(clause41)
                solver.add(clause42)
                solver.add(clause43)

            clause44 = Or(epsilon[i - 1], x[i - 1][self.adder_wordlength - 1], u[i - 1][self.adder_wordlength - 1], Not(z[i - 1][self.adder_wordlength - 1]))
            clause45 = Or(epsilon[i - 1], Not(x[i - 1][self.adder_wordlength - 1]), Not(u[i - 1][self.adder_wordlength - 1]), z[i - 1][self.adder_wordlength - 1])
            clause46 = Or(Not(epsilon[i - 1]), x[i - 1][self.adder_wordlength - 1], Not(u[i - 1][self.adder_wordlength - 1]), Not(z[i - 1][self.adder_wordlength - 1]))
            clause47 = Or(Not(epsilon[i - 1]), Not(x[i - 1][self.adder_wordlength - 1]), u[i - 1][self.adder_wordlength - 1], z[i - 1][self.adder_wordlength - 1])

            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)

        # right shift
        zeta = [[Bool(f'zeta_{i}_{k}', ctx=ctx) for k in range(self.adder_wordlength - 1)] for i in range(1, self.max_adder + 1)]

        for i in range(1, self.max_adder + 1):
            zeta_sum = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    clause48_1 = Or(Not(zeta[i - 1][k]), Not(z[i - 1][j + k]), c[i][j])
                    clause48_2 = Or(Not(zeta[i - 1][k]), z[i - 1][j + k], Not(c[i][j]))
                    # solver.add(And(clause48_1, clause48_2))
                    solver.add(clause48_1)
                    solver.add(clause48_2)

                zeta_sum.append(zeta[i - 1][k])

            solver.add(AtMost(*zeta_sum, 1))
            solver.add(AtLeast(*zeta_sum, 1))

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    clause49_1 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][self.adder_wordlength - 1]), c[i][self.adder_wordlength - 2 - b])
                    clause49_2 = Or(Not(zeta[i - 1][kf]), z[i - 1][self.adder_wordlength - 1], Not(c[i][self.adder_wordlength - 2 - b]))
                    solver.add(clause49_1)
                    solver.add(clause49_2)

                    clause50 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][b]))
                    solver.add(clause50)

            clause51_1 = Or(Not(z[i - 1][self.adder_wordlength - 1]), c[i][self.adder_wordlength - 1])
            clause51_2 = Or(z[i - 1][self.adder_wordlength - 1], Not(c[i][self.adder_wordlength - 1]))
            # solver.add(And(clause51_1, clause51_2))
            solver.add(clause51_1)
            solver.add(clause51_2)

        # set connected coefficient
        connected_coefficient = half_order + 1 - self.avail_dsp

        # solver connection
        theta = [[Bool(f'theta_{i}_{m}', ctx=ctx) for m in range(half_order + 1)] for i in range(self.max_adder + 2)]
        iota = [Bool(f'iota_{m}', ctx=ctx) for m in range(half_order + 1)]
        t = [[Bool(f't_{m}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]

        iota_sum = []
        for m in range(half_order + 1):
            theta_or = []
            for i in range(self.max_adder + 2):
                for word in range(self.adder_wordlength):
                    clause52_1 = Or(Not(theta[i][m]), Not(iota[m]), Not(c[i][word]), t[m][word])
                    clause52_2 = Or(Not(theta[i][m]), Not(iota[m]), c[i][word], Not(t[m][word]))
                    solver.add(clause52_1)
                    solver.add(clause52_2)
                theta_or.append(theta[i][m])
            # print(f"theta or {theta_or}")
            solver.add(Or(*theta_or))

        for m in range(half_order + 1):
            iota_sum.append(iota[m])
        solver.add(AtMost(*iota_sum, connected_coefficient))
        solver.add(AtLeast(*iota_sum, connected_coefficient))

        # Left Shifter in result module
        # k is the shift selector
        o = [[Bool(f'o_{m}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
        phi = [[Bool(f'phi_{m}_{k}', ctx=ctx) for k in range(self.adder_wordlength - 1)] for m in range(half_order + 1)]

        for m in range(half_order + 1):
            phi_sum = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    clause53_1 = Or(Not(phi[m][k]), Not(t[m][j]), o[m][j + k])
                    clause53_2 = Or(Not(phi[m][k]), t[m][j], Not(o[m][j + k]))
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause53_1)
                    solver.add(clause53_2)

                phi_sum.append(phi[m][k])

            solver.add(AtMost(*phi_sum, 1))
            solver.add(AtLeast(*phi_sum, 1))

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    clause54 = Or(Not(phi[m][kf]), Not(o[m][b]))
                    clause55 = Or(Not(phi[m][kf]), Not(t[m][self.adder_wordlength - 1]), t[m][self.adder_wordlength - 2 - b])
                    clause56 = Or(Not(phi[m][kf]), t[m][self.adder_wordlength - 1], Not(t[m][self.adder_wordlength - 2 - b]))
                    solver.add(clause54)
                    solver.add(clause55)
                    solver.add(clause56)

            clause57_1 = Or(Not(t[m][self.adder_wordlength - 1]), o[m][self.adder_wordlength - 1])
            clause57_2 = Or(t[m][self.adder_wordlength - 1], Not(o[m][self.adder_wordlength - 1]))
            # solver.add(And(clause7_1, clause7_2))
            solver.add(clause57_1)
            solver.add(clause57_2)

        rho = [Bool(f'rho_{m}', ctx=ctx) for m in range(half_order + 1)]
        o_xor = [[Bool(f'o_xor_{m}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
        h_ext = [[Bool(f'h_ext_{m}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
        cout_res = [[Bool(f'cout_res_{m}_{w}', ctx=ctx) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]

        # xor
        for m in range(half_order + 1):
            for word in range(self.adder_wordlength):
                clause58_1 = Or(o[m][word], rho[m], Not(o_xor[m][word]))
                clause59_1 = Or(o[m][word], Not(rho[m]), o_xor[m][word])
                clause60_1 = Or(Not(o[m][word]), rho[m], o_xor[m][word])
                clause61_1 = Or(Not(o[m][word]), Not(rho[m]), Not(o_xor[m][word]))
                solver.add(clause58_1)
                solver.add(clause59_1)
                solver.add(clause60_1)
                solver.add(clause61_1)
        
        # ripple carry half adder
        for m in range(half_order + 1):
            clause62_1 = Or(o_xor[m][0], rho[m], Not(h_ext[m][0]))
            clause63_1 = Or(o_xor[m][0], Not(rho[m]), h_ext[m][0])
            clause64_1 = Or(Not(o_xor[m][0]), rho[m], h_ext[m][0])
            clause65_1 = Or(Not(o_xor[m][0]), Not(rho[m]), Not(h_ext[m][0]))
            solver.add(clause62_1)
            solver.add(clause63_1)
            solver.add(clause64_1)
            solver.add(clause65_1)

            clause66_1 = Or(o_xor[m][0], Not(cout_res[m][0]))
            clause67_1 = Or(Not(o_xor[m][0]), Not(rho[m]), cout_res[m][0]) 
            clause68_1 = Or(o_xor[m][0], rho[m], Not(cout_res[m][0])) 
            clause69_1 = Or(rho[m], Not(cout_res[m][0]))
            solver.add(clause66_1)
            solver.add(clause67_1)
            solver.add(clause68_1)
            solver.add(clause69_1)

            for word in range(1, self.adder_wordlength):
                clause70_1 = Or(o_xor[m][word], cout_res[m][word - 1], Not(h_ext[m][word]))
                clause71_1 = Or(o_xor[m][word], Not(cout_res[m][word - 1]), h_ext[m][word])
                clause72_1 = Or(Not(o_xor[m][word]), cout_res[m][word - 1], h_ext[m][word])
                clause73_1 = Or(Not(o_xor[m][word]), Not(cout_res[m][word - 1]), Not(h_ext[m][word]))
                solver.add(clause70_1)
                solver.add(clause71_1)
                solver.add(clause72_1)
                solver.add(clause73_1)

                clause74_1 = Or(o_xor[m][word], Not(cout_res[m][word]))
                clause75_1 = Or(Not(o_xor[m][word]), Not(cout_res[m][word - 1]), cout_res[m][word]) 
                clause76_1 = Or(o_xor[m][word], cout_res[m][word - 1], Not(cout_res[m][word])) 
                clause77_1 = Or(cout_res[m][word - 1], Not(cout_res[m][word]))
                solver.add(clause74_1)
                solver.add(clause75_1)
                solver.add(clause76_1)
                solver.add(clause77_1)

        #conenction h_ext to h
        for m in range(half_order + 1):
            for word in range(self.adder_wordlength):
                if word <= self.wordlength - 1:
                    clause58 = Or(h[m][word], Not(h_ext[m][word]))
                    clause59 = Or(Not(h[m][word]), h_ext[m][word])
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause58)
                    solver.add(clause59)
                else:
                    clause58 = Or(h[m][self.wordlength - 1], Not(h_ext[m][word]))
                    clause59 = Or(Not(h[m][self.wordlength - 1]), h_ext[m][word])
                    # solver.add(And(clause3_1, clause3_2))
                    solver.add(clause58)
                    solver.add(clause59)

        # adder depth constraint
        if self.adder_depth > 0:
            psi_alpha = [[Bool(f'psi_alpha_{i}_{d}', ctx=ctx) for d in range(self.adder_depth)] for i in range(1, self.max_adder + 1)]
            psi_beta = [[Bool(f'psi_beta_{i}_{d}', ctx=ctx) for d in range(self.adder_depth)] for i in range(1, self.max_adder + 1)]
            psi_alpha_sum = []
            psi_beta_sum = []

            for i in range(1, self.max_adder + 1):
                clause60 = Or(Not(psi_alpha[i - 1][0]), alpha[i - 1][0])
                clause61 = Or(Not(psi_beta[i - 1][0]), beta[i - 1][0])
                solver.add(clause60)
                solver.add(clause61)

                psi_alpha_sum.append(psi_alpha[i - 1][0])
                psi_beta_sum.append(psi_beta[i - 1][0])

                

                if self.adder_depth == 1:
                    continue

                for d in range(1, self.adder_depth):
                    for a in range(i - 1):
                        clause63 = Or(Not(psi_alpha[i - 1][d]), alpha[i - 1][a])
                        clause64 = Or(Not(psi_alpha[i - 1][d]), psi_alpha[a][d - 1])
                        solver.add(clause63)
                        solver.add(clause64)

                        clause65 = Or(Not(psi_beta[i - 1][d]), beta[i - 1][a])
                        clause66 = Or(Not(psi_beta[i - 1][d]), psi_beta[a][d - 1])
                        solver.add(clause65)
                        solver.add(clause66)

                    psi_alpha_sum.append(psi_alpha[i - 1][d])
                    psi_beta_sum.append(psi_beta[i - 1][d])
                
                solver.add(AtMost(*psi_alpha_sum, 1))
                solver.add(AtLeast(*psi_alpha_sum, 1))
                solver.add(AtMost(*psi_beta_sum, 1))
                solver.add(AtLeast(*psi_beta_sum, 1))
        
    
        
        start_time=time.time()
        
        print("solver running")


        satisfiability = 'unsat'

        if solver.check() == sat:
            
            end_time = time.time()


            satisfiability = 'sat'

            print("solver sat")
            model = solver.model()

            # Calculate and store h coefficients
            self.h_res = []
            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = model.eval(h[m][w], model_completion=True)
                    print(f"h{m}_{w} = ", bool_value)

                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * (1 if bool_value else 0)
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2 ** (w - self.fracW) * (1 if bool_value else 0)

                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ", self.h_res)

            # Store h coefficients in result_model
            self.result_model.update({"h_res": self.h_res})

            # Calculate and store gain coefficient
            gain_coef = 0
            for g in range(self.gain_wordlength):
                # Evaluate the boolean value from the model
                bool_value = model.eval(gain[g], model_completion=True)
                print(f"gain{g}= ", bool_value)

                # Convert boolean to integer (0 or 1) and calculate the term
                if g < self.gain_fracW:
                    gain_coef += 2 ** -(self.gain_fracW - g) * (1 if bool_value else 0)
                else:
                    gain_coef += 2 ** (g - self.gain_fracW) * (1 if bool_value else 0)

            self.gain_res = gain_coef
            print("gain Coeffs: ", self.gain_res)

            # Store gain coefficient
            self.result_model.update({"gain": self.gain_res})

             # Store h
            h_values = []
            for i in range(len(h)):
                h_row = []
                for a in range(len(h[i])):
                    value = 1 if model.eval(h[i][a], model_completion=True) else 0
                    h_row.append(value)
                h_values.append(h_row)
            self.result_model.update({"h": h_values})

            # Store alpha selectors
            alpha_values = []
            for i in range(len(alpha)):
                alpha_row = []
                for a in range(len(alpha[i])):
                    value = 1 if model.eval(alpha[i][a], model_completion=True) else 0
                    alpha_row.append(value)
                alpha_values.append(alpha_row)
            self.result_model.update({"alpha": alpha_values})

            # Store beta selectors
            beta_values = []
            for i in range(len(beta)):
                beta_row = []
                for a in range(len(beta[i])):
                    value = 1 if model.eval(beta[i][a], model_completion=True) else 0
                    beta_row.append(value)
                beta_values.append(beta_row)
            self.result_model.update({"beta": beta_values})

            # Store gamma (left shift selectors)
            gamma_values = []
            for i in range(len(gamma)):
                gamma_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if model.eval(gamma[i][k], model_completion=True) else 0
                    gamma_row.append(value)
                gamma_values.append(gamma_row)
            self.result_model.update({"gamma": gamma_values})

            # Store delta selectors
            delta_values = []
            for i in range(len(delta)):
                value = 1 if model.eval(delta[i], model_completion=True) else 0
                delta_values.append(value)
            self.result_model.update({"delta": delta_values})

            # Store epsilon selectors
            epsilon_values = []
            for i in range(len(epsilon)):
                value = 1 if model.eval(epsilon[i], model_completion=True) else 0
                epsilon_values.append(value)
            self.result_model.update({"epsilon": epsilon_values})

            # Store zeta (right shift selectors)
            zeta_values = []
            for i in range(len(zeta)):
                zeta_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if model.eval(zeta[i][k], model_completion=True) else 0
                    zeta_row.append(value)
                zeta_values.append(zeta_row)
            self.result_model.update({"zeta": zeta_values})

            # Store phi (left shift selectors in result)
            phi_values = []
            for i in range(len(phi)):
                phi_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if model.eval(phi[i][k], model_completion=True) else 0
                    phi_row.append(value)
                phi_values.append(phi_row)
            self.result_model.update({"phi": phi_values})

            # Store theta array
            theta_values = []
            for i in range(len(theta)):
                theta_row = []
                for m in range(half_order + 1):
                    value = 1 if model.eval(theta[i][m], model_completion=True) else 0
                    theta_row.append(value)
                theta_values.append(theta_row)
            self.result_model.update({"theta": theta_values})

            # Store iota array
            iota_values = []
            for m in range(len(iota)):
                value = 1 if model.eval(iota[m], model_completion=True) else 0
                iota_values.append(value)
            self.result_model.update({"iota": iota_values})

            # Store rho array
            rho_values = []
            for m in range(len(rho)):
                value = 1 if model.eval(rho[m], model_completion=True) else 0
                rho_values.append(value)
            self.result_model.update({"rho": rho_values})

            
            
                      

        else:
            # print(f"unsat core {solver.unsat_core()}")
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        # for item in self.result_model:
        #     print(f"result of {item} is {self.result_model[item]}")

        print(f"\n************Z3 Report****************")
        print(f"Total number of variables            : ")
        print(f"Total number of constraints (clauses): {len(solver.assertions())}\n" )

        return self.result_model , satisfiability


if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_current = 6
    accuracy = 3
    wordlength = 14
    gain_upperbound = 4
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 4

    adder_count = 6
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    
    

    space = order_current * accuracy
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.4*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 6
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -5
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    # Create FIRFilter instance
    fir_filter = FIRFilterZ3(
                 filter_type, 
                 order_current, 
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
                 7,
                 2
                 )

    # fir_filter.run_barebone(1,'try_h_zero_count',1)
    target_result = fir_filter.runsolver(1,10)
    print(target_result)
    
import numpy as np
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import time
from sat_variable_handler import VariableMapper
from pb2cnf import PB2CNF
from rat2bool import Rat2bool
import multiprocessing
from solver_func import SolverFunc



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
        
    

class FIRFilterPysat:
    def __init__(self, 
                 filter_type, 
                 order_current, 
                 freqx_axis, 
                 freq_upper, 
                 freq_lower, 
                 ignore_lowerbound, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 intW,
                 ):
        self.filter_type = filter_type
        self.order_current = order_current
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.max_adder = adder_count
        self.wordlength = wordlength
        self.adder_wordlength = self.wordlength + adder_wordlength_ext # New adder wordlength for bitshifting
        

        self.freq_upper_lin = 0
        self.freq_lower_lin = 0

        self.intW = intW
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound = gain_upperbound
        self.gain_lowerbound = gain_lowerbound

        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.result_model = {}


    def run_barebone(self, solver_id):
        half_order = (self.order_current // 2)

        sf = SolverFunc(self.filter_type, self.order_current)
        self.freq_upper_lin = [f if not np.isnan(f) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [f if not np.isnan(f) else np.nan for f in self.freq_lower]

        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound, dtype=float)
        self.ignore_lowerbound = sf.db_to_linear(self.ignore_lowerbound_np)
        
        var_mapper = VariableMapper(half_order, self.wordlength,self.adder_wordlength, self.max_adder, self.adder_depth)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)
        
        top_var = var_mapper.max_int_value
        pb2cnf = PB2CNF(top_var)
        r2b = Rat2bool()

        solver_list = ['cadical103', 'cadical153','cadical195','glucose421','glucose41','minisat-gh','minisat22','lingeling','gluecard30','glucose30','gluecard41','maplesat']
        solver = Solver(name=solver_list[solver_id])

        #bound the gain to upper and lowerbound
        gain_literals = []

        for g in range(self.wordlength):
            gain_literals.append(v2i(('gain', g)))

        #round it first to the given wordlength
        self.gain_upperbound = r2b.frac2round([self.gain_upperbound],self.wordlength,self.fracW)[0]
        self.gain_lowerbound = r2b.frac2round([self.gain_lowerbound],self.wordlength,self.fracW)[0]
        
        #weight is 1, because it is multiplied to nothing, lits is 2d thus the bracket
        gain_weight = [1]
        cnf1 = pb2cnf.atleast(gain_weight,[gain_literals],self.gain_lowerbound,self.fracW)
        for clause in cnf1:
            solver.add_clause(clause)

        cnf2 = pb2cnf.atmost(gain_weight,[gain_literals],self.gain_upperbound,self.fracW)
        for clause in cnf2:
            solver.add_clause(clause)


        filter_literals = []
        filter_weights = []

        gain_freq_upper_prod_weights = []
        gain_freq_lower_prod_weights = []

        gain_upper_literals = []
        gain_lower_literals = []

        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue

            gain_literals.clear()
            filter_literals.clear()
            filter_weights.clear()

            gain_freq_upper_prod_weights.clear()
            gain_freq_lower_prod_weights.clear()
            
            gain_upper_literals.clear()
            gain_lower_literals.clear()
            

            for m in range(half_order + 1):
                filter_literals_temp = []
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    h_var = v2i(('h', m, w))
                    filter_literals_temp.append(h_var)
                filter_literals.append(filter_literals_temp)
                filter_weights.append(cm)

            #gain starts here
            gain_upper_literals_temp = []
            gain_lower_literals_temp = []

            #gain upperbound
            gain_upper_prod = -self.freq_upper_lin[omega]
            gain_freq_upper_prod_weights.append(gain_upper_prod)


            #declare the lits for pb2cnf
            if self.freq_lower_lin[omega] < self.ignore_lowerbound:
                gain_lower_prod = self.freq_upper_lin[omega]
            else:
                gain_lower_prod = -self.freq_lower_lin[omega]

            gain_freq_lower_prod_weights.append(gain_lower_prod)

            for g in range(self.wordlength):
                gain_var = v2i(('gain', g))
                gain_upper_literals_temp.append(gain_var)
                gain_lower_literals_temp.append(gain_var)

            gain_upper_literals.append(gain_upper_literals_temp)
            gain_lower_literals.append(gain_lower_literals_temp)
            

            #generate cnf for upperbound
            filter_upper_pb_weights = filter_weights + gain_freq_upper_prod_weights
            filter_upper_pb_weights = r2b.frac2round(filter_upper_pb_weights,self.wordlength,self.fracW)

            filter_upper_pb_literals = filter_literals + gain_upper_literals


            cnf3 = pb2cnf.atmost(weight=filter_upper_pb_weights,lits=filter_upper_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf3:
                solver.add_clause(clause)


            #generate cnf for lowerbound
            filter_lower_pb_weights = filter_weights + gain_freq_lower_prod_weights
    

            filter_lower_pb_weights = r2b.frac2round(filter_lower_pb_weights,self.wordlength,self.fracW)

            filter_lower_pb_literals = filter_literals + gain_lower_literals
            
            if len(filter_lower_pb_weights) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")
            
            cnf4 = pb2cnf.atleast(weight=filter_lower_pb_weights,lits=filter_lower_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf4:
                solver.add_clause(clause)
            
        

        print(f"Pysat: Barebone running with solver {solver_list[solver_id]}")

        satifiability = 'unsat'

        if solver.solve():
            satifiability = 'sat'
            self.model = solver.get_model()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    var_index = v2i(('h', m, w)) - 1
                    bool_value = self.model[var_index] > 0  # Convert to boolean

                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * bool_value
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fracW) * bool_value
                self.h_res.append(fir_coef)

            print(f"fir coeffs: {self.h_res}")

            gain_coef = 0
            for g in range(self.wordlength):
                var_index = v2i(('gain', g)) - 1
                bool_value = self.model[var_index] > 0  # Convert to boolean
                if g < self.fracW:
                    gain_coef += 2 ** -(self.fracW - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fracW) * bool_value

            self.gain_res = gain_coef


        else:
            print("Unsatisfiable")

        print(f"Pysat: Barebone done with solver {solver_list[solver_id]}")

        
        return satifiability,self.h_res,self.gain_res
       

    def runsolver_internal(self):
        half_order = (self.order_current // 2)
        
        print("Pysat solver called")
        var_mapper = VariableMapper(half_order, self.wordlength,self.adder_wordlength, self.max_adder, self.adder_depth)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)
        
        top_var = var_mapper.max_int_value
        pb2cnf = PB2CNF(top_var)
        r2b = Rat2bool()

        solver = Solver(name='cadical103')

        #bound the gain to upper and lowerbound
        gain_literals = []

        for g in range(self.wordlength):
            gain_literals.append(v2i(('gain', g)))

        #round it first to the given wordlength
        self.gain_upperbound = r2b.frac2round([self.gain_upperbound],self.wordlength,self.fracW)[0]
        self.gain_lowerbound = r2b.frac2round([self.gain_lowerbound],self.wordlength,self.fracW)[0]
        
        #weight is 1, because it is multiplied to nothing, lits is 2d thus the bracket
        gain_weight = [1]
        cnf1 = pb2cnf.atleast(gain_weight,[gain_literals],self.gain_lowerbound,self.fracW)
        for clause in cnf1:
            solver.add_clause(clause)

        cnf2 = pb2cnf.atmost(gain_weight,[gain_literals],self.gain_upperbound,self.fracW)
        for clause in cnf2:
            solver.add_clause(clause)


        filter_literals = []
        filter_weights = []

        gain_freq_upper_prod_weights = []
        gain_freq_lower_prod_weights = []

        gain_upper_literals = []
        gain_lower_literals = []

        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue

            gain_literals.clear()
            filter_literals.clear()
            filter_weights.clear()

            gain_freq_upper_prod_weights.clear()
            gain_freq_lower_prod_weights.clear()
            
            gain_upper_literals.clear()
            gain_lower_literals.clear()
            

            for m in range(half_order + 1):
                filter_literals_temp = []
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    h_var = v2i(('h', m, w))
                    filter_literals_temp.append(h_var)
                filter_literals.append(filter_literals_temp)
                filter_weights.append(cm)

            #gain starts here
            gain_upper_literals_temp = []
            gain_lower_literals_temp = []

            #gain upperbound
            gain_upper_prod = -self.freq_upper_lin[omega].item()
            gain_freq_upper_prod_weights.append(gain_upper_prod)

            #gain lowerbound

            #declare the lits for pb2cnf
            if self.freq_lower_lin[omega] < self.ignore_lowerbound:
                gain_lower_prod = self.freq_upper_lin[omega].item()
                # print("ignored ", self.freq_lower_lin[omega], " in frequency = ", self.freqx_axis[omega])
            else:
                gain_lower_prod = -self.freq_lower_lin[omega].item()

            gain_freq_lower_prod_weights.append(gain_lower_prod)

            for g in range(self.wordlength):
                gain_var = v2i(('gain', g))
                gain_upper_literals_temp.append(gain_var)
                gain_lower_literals_temp.append(gain_var)

            gain_upper_literals.append(gain_upper_literals_temp)
            gain_lower_literals.append(gain_lower_literals_temp)
            

            #generate cnf for upperbound
            filter_upper_pb_weights = filter_weights + gain_freq_upper_prod_weights
            filter_upper_pb_weights = r2b.frac2round(filter_upper_pb_weights,self.wordlength,self.fracW)

            filter_upper_pb_literals = filter_literals + gain_upper_literals

            if len(filter_upper_pb_weights) != len(filter_upper_pb_literals):
                raise Exception("sumtin wong with lower filter pb")

            # print("weight up: ",filter_upper_pb_weights)
            # print("lit up: ",filter_upper_pb_literals)

            cnf3 = pb2cnf.atmost(weight=filter_upper_pb_weights,lits=filter_upper_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf3:
                solver.add_clause(clause)


            #generate cnf for lowerbound
            filter_lower_pb_weights = filter_weights + gain_freq_lower_prod_weights
    

            filter_lower_pb_weights = r2b.frac2round(filter_lower_pb_weights,self.wordlength,self.fracW)

            filter_lower_pb_literals = filter_literals + gain_lower_literals
            
            if len(filter_lower_pb_weights) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")
            
            cnf4 = pb2cnf.atleast(weight=filter_lower_pb_weights,lits=filter_lower_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf4:
                solver.add_clause(clause)
            
        

        # Bitshift SAT starts here

        # c0,w is all 0 except 1, so input is 1
        for w in range(self.fracW+1, self.adder_wordlength):
            solver.add_clause([-v2i(('c', 0, w))])

        for w in range(self.fracW):
            solver.add_clause([-v2i(('c', 0, w))])

        solver.add_clause([v2i(('c', 0, self.fracW))])
        
        for i in range(1,self.max_adder+1):
            # Bound ci,0 to be odd number 
            solver.add_clause([v2i(('c', i, 0))])

        #last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            solver.add_clause([-v2i(('c', self.max_adder+1, w))])

            
        alpha_lits = []
        beta_lits = []

        # Input multiplexer
        for i in range(1, self.max_adder + 1):
            alpha_lits.clear()
            beta_lits.clear()
            for a in range(i):
                for word in range(self.adder_wordlength):
                    solver.add_clause([-v2i(('alpha', i, a)), -v2i(('c', a, word)), v2i(('l', i, word))])
                    solver.add_clause([-v2i(('alpha', i, a)), v2i(('c', a, word)), -v2i(('l', i, word))])
                    solver.add_clause([-v2i(('beta', i, a)), -v2i(('c', a, word)), v2i(('r', i, word))])
                    solver.add_clause([-v2i(('beta', i, a)), v2i(('c', a, word)), -v2i(('r', i, word))])

                alpha_lits.append(v2i(('alpha', i, a)))
                beta_lits.append(v2i(('beta', i, a)))

            cnf5 = pb2cnf.equal_card_one(alpha_lits)
            for clause in cnf5:
                solver.add_clause(clause)

            cnf6 = pb2cnf.equal_card_one(beta_lits)
            for clause in cnf6:
                solver.add_clause(clause)

        gamma_lits = []
        # Left Shifter
        for i in range(1, self.max_adder + 1):
            gamma_lits.clear()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    solver.add_clause([-v2i(('gamma', i, k)), -v2i(('l', i, j)), v2i(('s', i, j + k))])
                    solver.add_clause([-v2i(('gamma', i, k)), v2i(('l', i, j)), -v2i(('s', i, j + k))])

                gamma_lits.append(v2i(('gamma', i, k)))
            
            cnf7 = pb2cnf.equal_card_one(gamma_lits)
            for clauses in cnf7:
                solver.add_clause(clauses)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('s', i, b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('l', i, self.adder_wordlength - 1)), v2i(('l', i, self.adder_wordlength - 2 - b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), v2i(('l', i, self.adder_wordlength - 1)), -v2i(('l', i, self.adder_wordlength - 2 - b))])

            solver.add_clause([-v2i(('l', i, self.adder_wordlength - 1)), v2i(('s', i, self.adder_wordlength - 1))])
            solver.add_clause([v2i(('l', i, self.adder_wordlength - 1)), -v2i(('s', i, self.adder_wordlength - 1))])
        
            
        #delta selector
        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                solver.add_clause([-v2i(('delta', i)), -v2i(('s', i, word)), v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('s', i, word)), -v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), -v2i(('r', i, word)), v2i(('u', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('r', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('s', i, word)), v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('s', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('r', i, word)), v2i(('x', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('r', i, word)), -v2i(('x', i, word))])


        for i in range(1, self.max_adder + 1):
            for word in range(self.adder_wordlength):
                solver.add_clause([v2i(('u', i, word)), v2i(('epsilon', i)), -v2i(('y', i, word))])
                solver.add_clause([v2i(('u', i, word)), -v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), -v2i(('epsilon', i)), -v2i(('y', i, word))])

        for i in range(1, self.max_adder + 1):
            # Clauses for sum = a ⊕ b ⊕ cin at 0
            solver.add_clause([v2i(('x', i, 0)), v2i(('y', i, 0)), v2i(('epsilon', i)), -v2i(('z', i, 0))])
            solver.add_clause([v2i(('x', i, 0)), v2i(('y', i, 0)), -v2i(('epsilon', i)), v2i(('z', i, 0))])
            solver.add_clause([v2i(('x', i, 0)), -v2i(('y', i, 0)), v2i(('epsilon', i)), v2i(('z', i, 0))])
            solver.add_clause([-v2i(('x', i, 0)), v2i(('y', i, 0)), v2i(('epsilon', i)), v2i(('z', i, 0))])
            solver.add_clause([-v2i(('x', i, 0)), -v2i(('y', i, 0)), -v2i(('epsilon', i)), v2i(('z', i, 0))])
            solver.add_clause([-v2i(('x', i, 0)), -v2i(('y', i, 0)), v2i(('epsilon', i)), -v2i(('z', i, 0))])
            solver.add_clause([-v2i(('x', i, 0)), v2i(('y', i, 0)), -v2i(('epsilon', i)), -v2i(('z', i, 0))])
            solver.add_clause([v2i(('x', i, 0)), -v2i(('y', i, 0)), -v2i(('epsilon', i)), -v2i(('z', i, 0))])

            # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
            solver.add_clause([-v2i(('x', i, 0)), -v2i(('y', i, 0)), v2i(('cout', i, 0))])
            solver.add_clause([v2i(('x', i, 0)), v2i(('y', i, 0)), -v2i(('cout', i, 0))])
            solver.add_clause([-v2i(('x', i, 0)), -v2i(('epsilon', i)), v2i(('cout', i, 0))])
            solver.add_clause([v2i(('x', i, 0)), v2i(('epsilon', i)), -v2i(('cout', i, 0))])
            solver.add_clause([-v2i(('y', i, 0)), -v2i(('epsilon', i)), v2i(('cout', i, 0))])
            solver.add_clause([v2i(('y', i, 0)), v2i(('epsilon', i)), -v2i(('cout', i, 0))])

            for kf in range(1, self.adder_wordlength):
                # Clauses for sum = a ⊕ b ⊕ cin at kf
                solver.add_clause([v2i(('x', i, kf)), v2i(('y', i, kf)), v2i(('cout', i, kf - 1)), -v2i(('z', i, kf))])
                solver.add_clause([v2i(('x', i, kf)), v2i(('y', i, kf)), -v2i(('cout', i, kf - 1)), v2i(('z', i, kf))])
                solver.add_clause([v2i(('x', i, kf)), -v2i(('y', i, kf)), v2i(('cout', i, kf - 1)), v2i(('z', i, kf))])
                solver.add_clause([-v2i(('x', i, kf)), v2i(('y', i, kf)), v2i(('cout', i, kf - 1)), v2i(('z', i, kf))])
                solver.add_clause([-v2i(('x', i, kf)), -v2i(('y', i, kf)), -v2i(('cout', i, kf - 1)), v2i(('z', i, kf))])
                solver.add_clause([-v2i(('x', i, kf)), -v2i(('y', i, kf)), v2i(('cout', i, kf - 1)), -v2i(('z', i, kf))])
                solver.add_clause([-v2i(('x', i, kf)), v2i(('y', i, kf)), -v2i(('cout', i, kf - 1)), -v2i(('z', i, kf))])
                solver.add_clause([v2i(('x', i, kf)), -v2i(('y', i, kf)), -v2i(('cout', i, kf - 1)), -v2i(('z', i, kf))])

                # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
                solver.add_clause([-v2i(('x', i, kf)), -v2i(('y', i, kf)), v2i(('cout', i, kf))])
                solver.add_clause([v2i(('x', i, kf)), v2i(('y', i, kf)), -v2i(('cout', i, kf))])
                solver.add_clause([-v2i(('x', i, kf)), -v2i(('cout', i, kf - 1)), v2i(('cout', i, kf))])
                solver.add_clause([v2i(('x', i, kf)), v2i(('cout', i, kf - 1)), -v2i(('cout', i, kf))])
                solver.add_clause([-v2i(('y', i, kf)), -v2i(('cout', i, kf - 1)), v2i(('cout', i, kf))])
                solver.add_clause([v2i(('y', i, kf)), v2i(('cout', i, kf - 1)), -v2i(('cout', i, kf))])

            solver.add_clause([v2i(('epsilon', i)), v2i(('x', i, self.adder_wordlength - 1)), v2i(('u', i, self.adder_wordlength - 1)), -v2i(('z', i, self.adder_wordlength - 1))])
            solver.add_clause([v2i(('epsilon', i)), -v2i(('x', i, self.adder_wordlength - 1)), -v2i(('u', i, self.adder_wordlength - 1)), v2i(('z', i, self.adder_wordlength - 1))])
            solver.add_clause([-v2i(('epsilon', i)), v2i(('x', i, self.adder_wordlength - 1)), -v2i(('u', i, self.adder_wordlength - 1)), -v2i(('z', i, self.adder_wordlength - 1))])
            solver.add_clause([-v2i(('epsilon', i)), -v2i(('x', i, self.adder_wordlength - 1)), v2i(('u', i, self.adder_wordlength - 1)), v2i(('z', i, self.adder_wordlength - 1))])
        
        zeta_lits = []
        for i in range(1, self.max_adder + 1):
            zeta_lits.clear()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    solver.add_clause([-v2i(('zeta', i, k)), -v2i(('z', i, j + k)), v2i(('c', i, j))])
                    solver.add_clause([-v2i(('zeta', i, k)), v2i(('z', i, j + k)), -v2i(('c', i, j))])

                zeta_lits.append(v2i(('zeta', i, k)))

            cnf8 = pb2cnf.equal_card_one(zeta_lits)

            for clauses in cnf8:
                solver.add_clause(clauses)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, self.adder_wordlength - 1)), v2i(('c', i, self.adder_wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), v2i(('z', i, self.adder_wordlength - 1)), -v2i(('c', i, self.adder_wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, b))])

            solver.add_clause([-v2i(('z', i, self.adder_wordlength - 1)), v2i(('c', i, self.adder_wordlength - 1))])
            solver.add_clause([v2i(('z', i, self.adder_wordlength - 1)), -v2i(('c', i, self.adder_wordlength - 1))])

            

        # Set connected coefficient
        connected_coefficient = half_order + 1 - self.avail_dsp

        # Solver connection (theta, iota, and t)
        theta_lits = []
        iota_lits = []

        for m in range(half_order + 1):
            theta_or = []
            for i in range(self.max_adder + 2):
                for word in range(self.adder_wordlength):
                    solver.add_clause([-v2i(('theta', i, m)), -v2i(('iota', m)), -v2i(('c', i, word)), v2i(('t', m, word))])
                    solver.add_clause([-v2i(('theta', i, m)), -v2i(('iota', m)), v2i(('c', i, word)), -v2i(('t', m, word))])

                theta_or.append(v2i(('theta', i, m)))
            
            # Ensure that at least one `theta[i][m]` is true per `i`
            solver.add_clause(theta_or)

        for m in range(half_order + 1):
            iota_lits.append(v2i(('iota', m)))
        
        # Ensure that exactly `connected_coefficient` number of `iota[m]` are true
        cnf_iota = pb2cnf.equal_card(iota_lits, connected_coefficient)
        
        for clause in cnf_iota:
            solver.add_clause(clause)

        # Left Shifter in the result module (phi, h_ext logic)
        phi_lits = []
        for m in range(half_order + 1):
            phi_lits.clear()
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    solver.add_clause([-v2i(('phi', m, k)), -v2i(('t', m, j)), v2i(('h_ext', m, j + k))])
                    solver.add_clause([-v2i(('phi', m, k)), v2i(('t', m, j)), -v2i(('h_ext', m, j + k))])

                phi_lits.append(v2i(('phi', m, k)))

            cnf_phi = pb2cnf.equal_card_one(phi_lits)
            for clause in cnf_phi:
                solver.add_clause(clause)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('phi', m, kf)), -v2i(('h_ext', m, b))])
                    solver.add_clause([-v2i(('phi', m, kf)), -v2i(('t', m, self.adder_wordlength - 1)), v2i(('t', m, self.adder_wordlength - 2 - b))])
                    solver.add_clause([-v2i(('phi', m, kf)), v2i(('t', m, self.adder_wordlength - 1)), -v2i(('t', m, self.adder_wordlength - 2 - b))])

            solver.add_clause([-v2i(('t', m, self.adder_wordlength - 1)), v2i(('h_ext', m, self.adder_wordlength - 1))])
            solver.add_clause([v2i(('t', m, self.adder_wordlength - 1)), -v2i(('h_ext', m, self.adder_wordlength - 1))])

        # Connect h_ext to h
        for m in range(half_order + 1):
            for word in range(self.adder_wordlength):
                if word <= self.wordlength - 1:
                    solver.add_clause([v2i(('h', m, word)), -v2i(('h_ext', m, word))])
                    solver.add_clause([-v2i(('h', m, word)), v2i(('h_ext', m, word))])
                else:
                    solver.add_clause([v2i(('h', m, self.wordlength - 1)), -v2i(('h_ext', m, word))])
                    solver.add_clause([-v2i(('h', m, self.wordlength - 1)), v2i(('h_ext', m, word))])

        # Adder depth constraint (psi_alpha, psi_beta logic)
        if self.adder_depth > 0:
            for i in range(1, self.max_adder + 1):
                psi_alpha_lits = []
                psi_beta_lits = []

                # Ensure psi_alpha[0] implies alpha[0] and psi_beta[0] implies beta[0]
                solver.add_clause([-v2i(('psi_alpha', i, 0)), v2i(('alpha', i, 0))])
                solver.add_clause([-v2i(('psi_beta', i, 0)), v2i(('beta', i, 0))])

                for d in range(self.adder_depth):
                    psi_alpha_lits.append(v2i(('psi_alpha', i, d)))
                    psi_beta_lits.append(v2i(('psi_beta', i, d)))

                cnf_psi_alpha = pb2cnf.equal_card_one(psi_alpha_lits)
                cnf_psi_beta = pb2cnf.equal_card_one(psi_beta_lits)

                for clause in cnf_psi_alpha:
                    solver.add_clause(clause)
                for clause in cnf_psi_beta:
                    solver.add_clause(clause)

                if self.adder_depth > 1:
                    for d in range(1, self.adder_depth):
                        for a in range(i - 1):
                            solver.add_clause([-v2i(('psi_alpha', i, d)), v2i(('alpha', i, a))])
                            solver.add_clause([-v2i(('psi_alpha', i, d)), v2i(('psi_alpha', i, d - 1))])

                            solver.add_clause([-v2i(('psi_beta', i, d)), v2i(('beta', i, a))])
                            solver.add_clause([-v2i(('psi_beta', i, d)), v2i(('psi_beta', i, d - 1))])

        print("solver running")
        start_time = time.time()

        satifiability = 'unsat'

        if solver.solve():
            satifiability = 'sat'
            print("solver sat")
            self.model = solver.get_model()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    var_index = v2i(('h', m, w)) - 1
                    bool_value = self.model[var_index] > 0  # Convert to boolean

                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * bool_value
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fracW) * bool_value
                self.h_res.append(fir_coef)
                self.result_model[f"h[{m}]"] = fir_coef
            print("FIR Coeffs calculated: ", self.h_res)

            gain_coef = 0
            for g in range(self.wordlength):
                var_index = v2i(('gain', g)) - 1
                bool_value = self.model[var_index] > 0  # Convert to boolean
                if g < self.fracW:
                    gain_coef += 2 ** -(self.fracW - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fracW) * bool_value

            self.gain_res = gain_coef
            self.result_model["gain"] = gain_coef
            print("gain Coeffs: ", self.gain_res)

           # Store alpha and beta selectors
            for i in range(1, self.max_adder + 1):
                for a in range(i):
                    self.result_model[f'alpha[{i}][{a}]'] = 1 if self.model[v2i(('alpha', i, a)) - 1] > 0 else 0
                for a in range(i):
                    self.result_model[f'beta[{i}][{a}]'] = 1 if self.model[v2i(('beta', i, a)) - 1] > 0 else 0

            # Store gamma (left shift selectors)
            for i in range(1, self.max_adder + 1):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'gamma[{i}][{k}]'] = 1 if self.model[v2i(('gamma', i, k)) - 1] > 0 else 0

            # Store delta selectors
            for i in range(1, self.max_adder + 1):
                self.result_model[f'delta[{i}]'] = 1 if self.model[v2i(('delta', i)) - 1] > 0 else 0

            # Store epsilon selectors
            for i in range(1, self.max_adder + 1):
                self.result_model[f'epsilon[{i}]'] = 1 if self.model[v2i(('epsilon', i)) - 1] > 0 else 0

            # Store zeta (right shift selectors)
            for i in range(1, self.max_adder + 1):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'zeta[{i}][{k}]'] = 1 if self.model[v2i(('zeta', i, k)) - 1] > 0 else 0

            # Store phi (left shift selectors in result)
            for i in range(half_order + 1):
                for k in range(self.adder_wordlength - 1):
                    self.result_model[f'phi[{i}][{k}]'] = 1 if self.model[v2i(('phi', i, k)) - 1] > 0 else 0

            # Store theta array
            for i in range(self.max_adder + 2):
                for m in range(half_order + 1):
                    self.result_model[f'theta[{i}][{m}]'] = 1 if self.model[v2i(('theta', i, m)) - 1] > 0 else 0

            # Store iota array
            for m in range(half_order + 1):
                self.result_model[f'iota[{m}]'] = 1 if self.model[v2i(('iota', m)) - 1] > 0 else 0

        else:
            print("Unsatisfiable")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")
        solver.delete()
        for item in self.result_model:
            print(f"result of {item} is {self.result_model[item]}")

        print(f"\n************PySAT Report****************")
        print(f"Total number of main variables       : {top_var}")
        print(f"Total number of auxiliary variables  : {pb2cnf.var-top_var}")
        print(f"Total number of variables            : {pb2cnf.var}")        
        print(f"Total number of constraints (clauses): {solver.nof_clauses()}\n" )

        return duration, satifiability


    

    def runsolver(self, timeout=0):
        if timeout == 0:
            return self.runsolver_internal()

        #added function to do timeout in PySat

        manager = multiprocessing.Manager()
        shared_h_res = manager.list()
        shared_gain_res = manager.list()
        shared_result = manager.dict()

        process = multiprocessing.Process(target=run_solver_wrapper, args=(shared_h_res, shared_gain_res, shared_result, self))
        process.start()

        process.join(timeout)

        if process.is_alive():
            print("Solver timed out! Terminating...")
            process.terminate()
            process.join()
            return timeout, 'timeout'

        self.h_res = list(shared_h_res)  
        self.gain_res = list(shared_gain_res) 

        duration = shared_result.get('duration', None)
        satisfiability = shared_result.get('satisfiability', 'unsat')
        self.model = shared_result.get('model', None)
        return duration, satisfiability

def run_solver_wrapper(shared_h_res, shared_gain_res, shared_result, fir_filter):
    duration, satisfiability = fir_filter.runsolver_internal()
    shared_h_res[:] = fir_filter.h_res
    shared_gain_res[:] = [fir_filter.gain_res]
    shared_result['duration'] = duration
    shared_result['satisfiability'] = satisfiability
    shared_result['model'] = fir_filter.model if satisfiability == 'sat' else None


if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 20
    accuracy = 1
    adder_count = 3
    wordlength = 10
    
    adder_depth = 2
    avail_dsp = 0
    adder_wordlength_ext = 2
    gain_upperbound = 3
    gain_lowerbound = 1
    intW = 4

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

    freq_upper[upper_half_point:end_point] = -40
    freq_lower[upper_half_point:end_point] = -1000


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterPysat(
                 filter_type, 
                 order_upper, 
                 freqx_axis, 
                 freq_upper, 
                 freq_lower, 
                 ignore_lowerbound, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 intW,
                 )

    # Run solver and plot result
    fir_filter.runsolver()

import numpy as np
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import time
import multiprocessing
import math
try:
    from .sat_variable_handler import VariableMapper
    from .pb2cnf import PB2CNF
    from .rat2bool import Rat2bool
    from .solver_func import SolverFunc
except: 
    from sat_variable_handler import VariableMapper
    from pb2cnf import PB2CNF
    from rat2bool import Rat2bool
    from solver_func import SolverFunc


class FIRFilterPysat:
    def __init__(self, 
                 filter_type, 
                 half_order, 
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
        self.half_order = half_order
        self.freqx_axis = freqx_axis
        self.upperbound = freq_upper
        self.lowerbound = freq_lower
        self.h_res = []
        self.gain_res = 0

        self.adder_count = adder_count
        self.wordlength = wordlength
        self.weight_wordlength = self.wordlength + 2 #increase this if you need higher intW for the weights

        self.adder_wordlength = self.wordlength + adder_wordlength_ext # New adder wordlength for bitshifting
        

        self.upperbound_lin = 0
        self.lowerbound_lin = 0

        self.intW = intW
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound = gain_upperbound
        self.gain_lowerbound = gain_lowerbound


        self.ignore_lowerbound_lin = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.result_model = {}

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        }

        return input_data_sf
    


    def run_barebone(self, solver_id, h_zero_count= 0):
        if h_zero_count == None:
            h_zero_count = 0
        self.h_res = []
        self.gain_res = 0
        self.model = []

        half_order = self.half_order - 1 #-1 is because i am lazy to change the code

        sf = SolverFunc(self.get_solver_func_dict())
        internal_upperbound_lin = [f if not np.isnan(f) else np.nan for f in self.upperbound]
        internal_lowerbound_lin = [f if not np.isnan(f) else np.nan for f in self.lowerbound]

        internal_ignore_lowerbound = self.ignore_lowerbound_lin

       
        # print(f"filter_type: {self.filter_type}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")

        
        var_mapper = VariableMapper(half_order, self.wordlength,self.adder_wordlength, self.adder_count, self.adder_depth, self.avail_dsp)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)
        
        top_var = var_mapper.max_int_value
        pb2cnf = PB2CNF(self.weight_wordlength, top_var)
        r2b = Rat2bool()

        solver_list = ['cadical103', 'cadical153','cadical195','glucose421','glucose41','minisat-gh','minisat22','lingeling','gluecard30','glucose30','gluecard41','maplesat']
        print(f"Pysat: Barebone running with solver {solver_list[solver_id]}")

        solver = Solver(name=solver_list[solver_id])

        #bound the gain to upper and lowerbound
        gain_literals = []

        for g in range(self.wordlength):
            gain_literals.append(v2i(('gain', g)))

        #round it first to the given wordlength
        self.gain_upperbound = r2b.frac2round([self.gain_upperbound],self.wordlength,self.fracW,True)[0]
        self.gain_lowerbound = r2b.frac2round([self.gain_lowerbound],self.wordlength,self.fracW,True)[0]
        
        #weight is 1, because it is multiplied to nothing, lits is 2d thus the bracket
        gain_weight = [1]
        cnf1 = pb2cnf.atleast(gain_weight,[gain_literals],self.gain_lowerbound,self.fracW)
        for clause in cnf1:
            solver.add_clause(clause)

        cnf2 = pb2cnf.atmost(gain_weight,[gain_literals],self.gain_upperbound,self.fracW)
        for clause in cnf2:
            solver.add_clause(clause)

        if h_zero_count > 0:
            print("pysat_barebone_running: try_h_zero_count")
            h_zero_sum = []
            for m in range(half_order+1):
                for word in range(self.wordlength):
                    solver.add_clause([-v2i(('h_zero', m)), -v2i(('h', m, word))])
                h_zero_sum.append(v2i(('h_zero', m)))

            cnf_h_zero = pb2cnf.equal_card(h_zero_sum, h_zero_count)
            
            for clause in cnf_h_zero:
                solver.add_clause(clause)
       


        filter_literals = []
        filter_weights = []

        gain_freq_upper_prod_weights = []
        gain_freq_lower_prod_weights = []

        gain_upper_literals = []
        gain_lower_literals = []

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
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
            gain_upper_prod = -internal_upperbound_lin[omega]
            gain_freq_upper_prod_weights.append(gain_upper_prod)


            #declare the lits for pb2cnf
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                gain_lower_prod = internal_upperbound_lin[omega]
            else:
                gain_lower_prod = -internal_lowerbound_lin[omega]

            gain_freq_lower_prod_weights.append(gain_lower_prod)

            for g in range(self.wordlength):
                gain_var = v2i(('gain', g))
                gain_upper_literals_temp.append(gain_var)
                gain_lower_literals_temp.append(gain_var)

            gain_upper_literals.append(gain_upper_literals_temp)
            gain_lower_literals.append(gain_lower_literals_temp)
            

            #generate cnf for upperbound
            filter_upper_pb_weights = filter_weights + gain_freq_upper_prod_weights
            filter_upper_pb_weights = r2b.frac2round(filter_upper_pb_weights,self.weight_wordlength,self.fracW,True)

            filter_upper_pb_literals = filter_literals + gain_upper_literals


            cnf3 = pb2cnf.atmost(weight=filter_upper_pb_weights,lits=filter_upper_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf3:
                solver.add_clause(clause)


            #generate cnf for lowerbound
            filter_lower_pb_weights = filter_weights + gain_freq_lower_prod_weights
    

            filter_lower_pb_weights = r2b.frac2round(filter_lower_pb_weights,self.weight_wordlength,self.fracW,True)

            filter_lower_pb_literals = filter_literals + gain_lower_literals
            
            if len(filter_lower_pb_weights) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")
            
            cnf4 = pb2cnf.atleast(weight=filter_lower_pb_weights,lits=filter_lower_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf4:
                solver.add_clause(clause)

        

            

        
        satisfiability = 'unsat'

        if solver.solve():
            satisfiability = 'sat'
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

        
        return satisfiability, self.h_res, self.gain_res
       

    def runsolver(self,solver_id,adderm,h_zero_count):
        self.adder_count = adderm
        self.h_res = []
        self.gain_res = 0
        half_order = self.half_order - 1 #-1 is because i am lazy to change the code
        
        print("Pysat solver called")
        var_mapper = VariableMapper(half_order, self.wordlength,self.adder_wordlength, self.adder_count, self.adder_depth, self.avail_dsp)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)
        
        top_var = var_mapper.max_int_value
        pb2cnf = PB2CNF(self.weight_wordlength, top_var)
        r2b = Rat2bool()

        sf = SolverFunc(self.get_solver_func_dict())
        internal_upperbound_lin = [f if not np.isnan(f) else np.nan for f in self.upperbound]
        internal_lowerbound_lin = [f if not np.isnan(f) else np.nan for f in self.lowerbound]

        internal_ignore_lowerbound = self.ignore_lowerbound_lin

        solver_list = ['cadical103', 'cadical153','cadical195','glucose421','glucose41','minisat-gh','minisat22','lingeling','gluecard30','glucose30','gluecard41','maplesat']
        print(f"Pysat: Barebone running with solver {solver_list[solver_id]}")

        solver = Solver(name=solver_list[solver_id])

        #bound the gain to upper and lowerbound
        gain_literals = []

        for g in range(self.wordlength):
            gain_literals.append(v2i(('gain', g)))

        #round it first to the given wordlength
        self.gain_upperbound = r2b.frac2round([self.gain_upperbound],self.wordlength,self.fracW,True)[0]
        self.gain_lowerbound = r2b.frac2round([self.gain_lowerbound],self.wordlength,self.fracW,True)[0]
        
        #weight is 1, because it is multiplied to nothing this is pure G<lb(G), lits is 2d thus the bracket
        gain_weight = [1]
        cnf1 = pb2cnf.atleast(gain_weight,[gain_literals],self.gain_lowerbound,self.fracW)
        for clause in cnf1:
            solver.add_clause(clause)

        cnf2 = pb2cnf.atmost(gain_weight,[gain_literals],self.gain_upperbound,self.fracW)
        for clause in cnf2:
            solver.add_clause(clause)

        if h_zero_count > 0:
            print("pysat_running: try_h_zero_count")
            h_zero_sum = []
            for m in range(half_order+1):
                for word in range(self.wordlength):
                    solver.add_clause([-v2i(('h_zero', m)), -v2i(('h', m, word))])
                h_zero_sum.append(v2i(('h_zero', m)))

            cnf_h_zero = pb2cnf.equal_card(h_zero_sum, h_zero_count)
            
            for clause in cnf_h_zero:
                solver.add_clause(clause)

        filter_literals = []
        filter_weights = []

        gain_freq_upper_prod_weights = []
        gain_freq_lower_prod_weights = []

        gain_upper_literals = []
        gain_lower_literals = []

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
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
            gain_upper_prod = -internal_upperbound_lin[omega].item()
            gain_freq_upper_prod_weights.append(gain_upper_prod)

            #gain lowerbound

            #declare the lits for pb2cnf
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                gain_lower_prod = internal_upperbound_lin[omega].item()
                # print("ignored ", internal_lowerbound_lin[omega], " in frequency = ", self.freqx_axis[omega])
            else:
                gain_lower_prod = -internal_lowerbound_lin[omega].item()

            gain_freq_lower_prod_weights.append(gain_lower_prod)

            for g in range(self.wordlength):
                gain_var = v2i(('gain', g))
                gain_upper_literals_temp.append(gain_var)
                gain_lower_literals_temp.append(gain_var)

            gain_upper_literals.append(gain_upper_literals_temp)
            gain_lower_literals.append(gain_lower_literals_temp)
            

            #generate cnf for upperbound
            filter_upper_pb_weights = filter_weights + gain_freq_upper_prod_weights
            filter_upper_pb_weights = r2b.frac2round(filter_upper_pb_weights,self.weight_wordlength,self.fracW,True)

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
    

            filter_lower_pb_weights = r2b.frac2round(filter_lower_pb_weights,self.weight_wordlength,self.fracW,True)

            filter_lower_pb_literals = filter_literals + gain_lower_literals
            
            if len(filter_lower_pb_weights) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")
            
            cnf4 = pb2cnf.atleast(weight=filter_lower_pb_weights,lits=filter_lower_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf4:
                solver.add_clause(clause)
            
        

        # dsp are odd numbers
        for k in range(self.adder_count + 2, self.avail_dsp+1):
            solver.add_clause([v2i(('c', k, 0))])

        
        # c0,w is always 0 except at 0 so input is 1
        for w in range(1, self.adder_wordlength):
            solver.add_clause([-v2i(('c', 0, w))])

        solver.add_clause([v2i(('c', 0, 0))])
    
        for i in range(1,self.adder_count+1):
            # Bound ci,0 to be odd number 
            solver.add_clause([v2i(('c', i, 0))])

        #last c or c[N+1] is connected to ground, so all zeroes
        for w in range(self.adder_wordlength):
            solver.add_clause([-v2i(('c', self.adder_count+1, w))])

            
        alpha_lits = []
        beta_lits = []

        # Input multiplexer
        for i in range(1, self.adder_count + 1):
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
        for i in range(1, self.adder_count + 1):
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
        for i in range(1, self.adder_count + 1):
            for word in range(self.adder_wordlength):
                solver.add_clause([-v2i(('delta', i)), -v2i(('s', i, word)), v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('s', i, word)), -v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), -v2i(('r', i, word)), v2i(('u', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('r', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('s', i, word)), v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('s', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('r', i, word)), v2i(('x', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('r', i, word)), -v2i(('x', i, word))])


        for i in range(1, self.adder_count + 1):
            for word in range(self.adder_wordlength):
                solver.add_clause([v2i(('u', i, word)), v2i(('epsilon', i)), -v2i(('y', i, word))])
                solver.add_clause([v2i(('u', i, word)), -v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), -v2i(('epsilon', i)), -v2i(('y', i, word))])

        for i in range(1, self.adder_count + 1):
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
        for i in range(1, self.adder_count + 1):
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

            


        # Solver connection (theta, iota, and t)

        for m in range(half_order + 1):
            theta_or = []
            for i in range(self.adder_count + 2 + self.avail_dsp):
                for word in range(self.adder_wordlength):
                    solver.add_clause([-v2i(('theta', i, m)), -v2i(('c', i, word)), v2i(('t', m, word))])
                    solver.add_clause([-v2i(('theta', i, m)), v2i(('c', i, word)), -v2i(('t', m, word))])

                theta_or.append(v2i(('theta', i, m)))
            
            # Ensure that at least one `theta[i][m]` is true per `i`
            solver.add_clause(theta_or)


        # Updated Left Shifter in the result module (including new variables and clauses)
        # Left Shifter in the result module (phi, h_ext, o, rho, o_xor, cout_res logic)
        for m in range(half_order + 1):
            phi_lits = []
            for k in range(self.adder_wordlength - 1):
                for j in range(self.adder_wordlength - 1 - k):
                    solver.add_clause([-v2i(('phi', m, k)), -v2i(('t', m, j)), v2i(('o', m, j + k))])
                    solver.add_clause([-v2i(('phi', m, k)), v2i(('t', m, j)), -v2i(('o', m, j + k))])
                phi_lits.append(v2i(('phi', m, k)))

            # Enforce that exactly one phi[m][k] is True
            cnf_phi = pb2cnf.equal_card_one(phi_lits)
            for clause in cnf_phi:
                solver.add_clause(clause)

            for kf in range(1, self.adder_wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('phi', m, kf)), -v2i(('o', m, b))])
                    solver.add_clause([-v2i(('phi', m, kf)), -v2i(('t', m, self.adder_wordlength - 1)), v2i(('t', m, self.adder_wordlength - 2 - b))])
                    solver.add_clause([-v2i(('phi', m, kf)), v2i(('t', m, self.adder_wordlength - 1)), -v2i(('t', m, self.adder_wordlength - 2 - b))])

            solver.add_clause([-v2i(('t', m, self.adder_wordlength - 1)), v2i(('o', m, self.adder_wordlength - 1))])
            solver.add_clause([v2i(('t', m, self.adder_wordlength - 1)), -v2i(('o', m, self.adder_wordlength - 1))])
        
        #xor
        for m in range(half_order + 1):
            for w in range(self.adder_wordlength):
                solver.add_clause([v2i(('o', m, w)), v2i(('rho', m)), -v2i(('o_xor', m, w))])
                solver.add_clause([v2i(('o', m, w)), -v2i(('rho', m)), v2i(('o_xor', m, w))])
                solver.add_clause([-v2i(('o', m, w)), v2i(('rho', m)), v2i(('o_xor', m, w))])
                solver.add_clause([-v2i(('o', m, w)), -v2i(('rho', m)), -v2i(('o_xor', m, w))])
        
        #ripple carry half adder 
        for m in range(half_order + 1):
            # Ripple carry half adder for w = 0
            solver.add_clause([v2i(('o_xor', m, 0)), v2i(('rho', m)), -v2i(('h_ext', m, 0))])
            solver.add_clause([v2i(('o_xor', m, 0)), -v2i(('rho', m)), v2i(('h_ext', m, 0))])
            solver.add_clause([-v2i(('o_xor', m, 0)), v2i(('rho', m)), v2i(('h_ext', m, 0))])
            solver.add_clause([-v2i(('o_xor', m, 0)), -v2i(('rho', m)), -v2i(('h_ext', m, 0))])

            # Carry out for w = 0
            solver.add_clause([v2i(('o_xor', m, 0)), -v2i(('cout_res', m, 0))])
            solver.add_clause([-v2i(('o_xor', m, 0)), -v2i(('rho', m)), v2i(('cout_res', m, 0))])
            solver.add_clause([v2i(('o_xor', m, 0)), v2i(('rho', m)), -v2i(('cout_res', m, 0))])
            solver.add_clause([v2i(('rho', m)), -v2i(('cout_res', m, 0))])

            # Ripple carry half adder for w > 0
            for w in range(1, self.adder_wordlength):
                solver.add_clause([v2i(('o_xor', m, w)), v2i(('cout_res', m, w - 1)), -v2i(('h_ext', m, w))])
                solver.add_clause([v2i(('o_xor', m, w)), -v2i(('cout_res', m, w - 1)), v2i(('h_ext', m, w))])
                solver.add_clause([-v2i(('o_xor', m, w)), v2i(('cout_res', m, w - 1)), v2i(('h_ext', m, w))])
                solver.add_clause([-v2i(('o_xor', m, w)), -v2i(('cout_res', m, w - 1)), -v2i(('h_ext', m, w))])

                # Carry out bits
                solver.add_clause([v2i(('o_xor', m, w)), -v2i(('cout_res', m, w))])
                solver.add_clause([-v2i(('o_xor', m, w)), -v2i(('cout_res', m, w - 1)), v2i(('cout_res', m, w))])
                solver.add_clause([v2i(('o_xor', m, w)), v2i(('cout_res', m, w - 1)), -v2i(('cout_res', m, w))])
                solver.add_clause([v2i(('cout_res', m, w - 1)), -v2i(('cout_res', m, w))])

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
            for i in range(1, self.adder_count + 1):
                psi_alpha_lits = []
                psi_beta_lits = []

                # Ensure psi_alpha[0] implies alpha[0] and psi_beta[0] implies beta[0]
                solver.add_clause([-v2i(('psi_alpha', i, 0)), -v2i(('alpha', i, 0))])
                solver.add_clause([-v2i(('psi_beta', i, 0)), v2i(('beta', i, 0))])

                psi_alpha_lits.append(v2i(('psi_alpha', i, 0)))
                psi_beta_lits.append(v2i(('psi_beta', i, 0)))

                if self.adder_depth == 1:
                    cnf_psi_alpha = pb2cnf.equal_card_one(psi_alpha_lits)
                    cnf_psi_beta = pb2cnf.equal_card_one(psi_beta_lits)

                    for clause in cnf_psi_alpha:
                        solver.add_clause(clause)
                    for clause in cnf_psi_beta:
                        solver.add_clause(clause)
                    continue

                for d in range(1, self.adder_depth):
                    for a in range(1, i):
                        solver.add_clause([-v2i(('psi_alpha', i, d)), -v2i(('alpha', i, a)), v2i(('psi_alpha', a, d - 1))])
                        psi_alpha_or = []
                        for j in range(d):
                            psi_alpha_or.append(v2i(('psi_alpha', a, j)))
                        solver.add_clause([-v2i(('psi_alpha', i, d)),-v2i(('alpha', i, a))] + psi_alpha_or)
                                                  
                        solver.add_clause([-v2i(('psi_beta', i, d)), -v2i(('beta', i, a)), v2i(('psi_beta', a, d - 1))])
                        psi_beta_or = []
                        for j in range(d):
                            psi_beta_or.append(v2i(('psi_beta', a, j)))
                        solver.add_clause([-v2i(('psi_beta', i, d)), -v2i(('psi_beta', a, d - 1))]+ psi_beta_or)

                    psi_alpha_lits.append(v2i(('psi_alpha', i, d)))
                    psi_beta_lits.append(v2i(('psi_beta', i, d)))

                
                cnf_psi_alpha = pb2cnf.equal_card_one(psi_alpha_lits)
                cnf_psi_beta = pb2cnf.equal_card_one(psi_beta_lits)

                for clause in cnf_psi_alpha:
                    solver.add_clause(clause)
                for clause in cnf_psi_beta:
                    solver.add_clause(clause)

        print("solver running")
        start_time = time.time()

        satisfiability = 'unsat'

        if solver.solve():
            end_time = time.time()

            satisfiability = 'sat'
            print("solver sat")
            self.model = solver.get_model()

            # Calculate h coefficients
            self.h_res = []
            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    var_index = v2i(('h', m, w)) - 1
                    bool_value = 1 if self.model[var_index] > 0 else 0  # Convert to boolean

                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * bool_value
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fracW) * bool_value
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ", self.h_res)

            # Store h coefficients
            self.result_model.update({"h_res": self.h_res})

            # Calculate gain coefficient
            gain_coef = 0
            for g in range(self.wordlength):
                var_index = v2i(('gain', g)) - 1
                bool_value = 1 if self.model[var_index] > 0 else 0  # Convert to boolean
                if g < self.fracW:
                    gain_coef += 2 ** -(self.fracW - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fracW) * bool_value

            self.gain_res = gain_coef
            print("gain Coeffs: ", self.gain_res)
            self.result_model.update({"gain": gain_coef})

             # Store h
            h_values = []
            for i in range(half_order + 1):
                h_row = []
                for a in range(self.wordlength):
                    var_index = v2i(('h', i, a)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    h_row.append(value)
                h_values.append(h_row)
            self.result_model.update({"h": h_values})

            # Store alpha selectors
            alpha_values = []
            for i in range(1, self.adder_count + 1):
                alpha_row = []
                for a in range(i):
                    var_index = v2i(('alpha', i, a)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    alpha_row.append(value)
                alpha_values.append(alpha_row)
            self.result_model.update({"alpha": alpha_values})

            # Store beta selectors
            beta_values = []
            for i in range(1, self.adder_count + 1):
                beta_row = []
                for a in range(i):
                    var_index = v2i(('beta', i, a)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    beta_row.append(value)
                beta_values.append(beta_row)
            self.result_model.update({"beta": beta_values})

            # Store gamma (left shift selectors)
            gamma_values = []
            for i in range(1, self.adder_count + 1):
                gamma_row = []
                for k in range(self.adder_wordlength - 1):
                    var_index = v2i(('gamma', i, k)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    gamma_row.append(value)
                gamma_values.append(gamma_row)
            self.result_model.update({"gamma": gamma_values})

            # Store delta selectors
            delta_values = []
            for i in range(1, self.adder_count + 1):
                var_index = v2i(('delta', i)) - 1
                value = 1 if self.model[var_index] > 0 else 0
                delta_values.append(value)
            self.result_model.update({"delta": delta_values})

            # Store epsilon selectors
            epsilon_values = []
            for i in range(1, self.adder_count + 1):
                var_index = v2i(('epsilon', i)) - 1
                value = 1 if self.model[var_index] > 0 else 0
                epsilon_values.append(value)
            self.result_model.update({"epsilon": epsilon_values})

            # Store zeta (right shift selectors)
            zeta_values = []
            for i in range(1, self.adder_count + 1):
                zeta_row = []
                for k in range(self.adder_wordlength - 1):
                    var_index = v2i(('zeta', i, k)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    zeta_row.append(value)
                zeta_values.append(zeta_row)
            self.result_model.update({"zeta": zeta_values})

            # Store phi (left shift selectors in result)
            phi_values = []
            for i in range(half_order + 1):
                phi_row = []
                for k in range(self.adder_wordlength - 1):
                    var_index = v2i(('phi', i, k)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    phi_row.append(value)
                phi_values.append(phi_row)
            self.result_model.update({"phi": phi_values})

            # Store theta array
            theta_values = []
            for i in range(self.adder_count + 2 + self.avail_dsp):
                theta_row = []
                for m in range(half_order + 1):
                    var_index = v2i(('theta', i, m)) - 1
                    value = 1 if self.model[var_index] > 0 else 0
                    theta_row.append(value)
                theta_values.append(theta_row)
            self.result_model.update({"theta": theta_values})          

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

        return self.result_model , satisfiability



    

    


if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    half_order = 5
    accuracy = 2
    wordlength = 13
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 3
    intW = 2

    adder_count = 6
    adder_depth = 3
    avail_dsp = 1
    adder_wordlength_ext = 2
    h_zero_count = 0	
    
    
    delta = 0.4
    passband_error = delta
    stopband_error = delta
    space = half_order * accuracy
    # Initialize freq_upper and freq_lower with NaN valuess
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -60

    #linearize the bound
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)


    # Create FIRFilter instance
    fir_filter = FIRFilterPysat(
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
                 intW,
                 )

    # Run solver and plot result
    target_result = fir_filter.runsolver(0,adder_count,h_zero_count)
    print(target_result)
    # fir_filter.run_barebone(0,5)

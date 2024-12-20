import numpy as np
from pysat.solvers import Solver
from pysat.pb import PBEnc
from pysat.formula import IDPool
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
        self.ih = 2
        self.fh = self.wordlength - self.ih

        self.gain_wordlength = 6  # 9 bits wordlength for gain
        self.ig = 3
        self.fg = self.gain_wordlength - self.ig

        self.gain_upperbound = 1.4
        self.gain_lowerbound = 1
        self.gain_bound_accuracy = 2  # 2 floating points

        self.ignore_lowerbound_lin = ignore_lowerbound_lin * (10 ** self.coef_accuracy) * (2 ** self.fh)

    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)

        encoding_type = 2
        
        print("solver called")
        sf = SolverFunc(self.filter_type, self.order_current)
        var_mapper = VariableMapper(half_order, self.wordlength, self.gain_wordlength, self.N)

        max_var = var_mapper.max_int_value

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)

        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound_lin)

        self.freq_upper_lin = [int((sf.db_to_linear(f)) * (10 ** self.coef_accuracy) * (2 ** self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * (10 ** self.coef_accuracy) * (2 ** self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]

        solver = Solver(name='minisat22')
        pool = IDPool(start_from=max_var)  # Ensure pool starts from some max value

        gain_coeffs = []
        gain_literals = []

        self.gain_upperbound_int = int(self.gain_upperbound * 2 ** self.fg * (10 ** self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound * 2 ** self.fg * (10 ** self.gain_bound_accuracy))

        for g in range(self.gain_wordlength):
            gain_coeffs.append((2 ** g) * (10 ** self.gain_bound_accuracy))
            gain_literals.append(v2i(('gain', g)))
            print("gain coef :", v2i(('gain', g)))

        print("gain coef: ",gain_coeffs)
        
        if self.gain_lowerbound_int > sum(gain_coeffs):
            raise("gain lowerbound is bigger than gain_coef")
        
        cnf = PBEnc.atmost(lits=gain_literals, weights=gain_coeffs, bound=self.gain_upperbound_int, vpool=pool, encoding=encoding_type)
        cnf.extend(PBEnc.atleast(lits=gain_literals, weights=gain_coeffs, bound=self.gain_lowerbound_int, vpool=pool, encoding=encoding_type).clauses)

        for clause in cnf.clauses:
            solver.add_clause(clause)

        filter_literals = []
        filter_coeffs = []
        gain_freq_upper_prod_coeffs = []
        gain_freq_lower_prod_coeffs = []

        filter_overflow_literals = []
        filter_overflow_coeffs = []

        gain_upper_overflow_literals = []
        gain_upper_overflow_coeffs = []

        gain_lower_overflow_literals = []
        gain_lower_overflow_coeffs = []

        gain_upper_literals = []
        gain_lower_literals = []

        max_positive_int_pbfunc = 2147483647
        max_negative_int_pbfunc = -2147483648

        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue

            gain_literals.clear()
            filter_literals.clear()
            filter_coeffs.clear()

            gain_freq_upper_prod_coeffs.clear()
            gain_freq_lower_prod_coeffs.clear()

            filter_overflow_literals.clear()
            filter_overflow_coeffs.clear()

            gain_upper_overflow_literals.clear()
            gain_upper_overflow_coeffs.clear()

            gain_lower_overflow_literals.clear()
            gain_lower_overflow_coeffs.clear()
            
            gain_upper_literals.clear()
            gain_lower_literals.clear()
            

            for m in range(half_order + 1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    h_var = v2i(('h', m, w))
                    if w == self.wordlength - 1:
                        cm_word_prod = int(cm * (10 ** self.coef_accuracy) * (-1*2 ** w) * (2 ** self.fg))
                    else:
                        cm_word_prod = int(cm * (10 ** self.coef_accuracy) * (2 ** w) * 2 ** self.fg)

                    if cm_word_prod > max_positive_int_pbfunc or cm_word_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(cm_word_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, h_var)
                        filter_overflow_literals.extend(overflow[0])
                        filter_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the product of cm: appended this to the sum coeff:", overflow[1], " with literal: ", overflow[0])
                    else:
                        filter_coeffs.append(cm_word_prod)
                        filter_literals.append(h_var)
            
            for g in range(self.gain_wordlength):
                gain_var = v2i(('gain', g))
                gain_upper_prod = int((-1* 2 ** g) * self.freq_upper_lin[omega])
                if gain_upper_prod > max_positive_int_pbfunc or gain_upper_prod < max_negative_int_pbfunc:
                    overflow = sf.overflow_handler(gain_upper_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain_var)
                    gain_upper_overflow_literals.extend(overflow[0])
                    gain_upper_overflow_coeffs.extend(overflow[1])
                    print("overflow happened in the gain upper product: appended this to the sum coeff:", overflow[1], " with literal: ", overflow[0])
                else:
                    gain_freq_upper_prod_coeffs.append(gain_upper_prod)
                    gain_upper_literals.append(gain_var)

                if self.freq_lower_lin[omega] < self.ignore_lowerbound_lin:
                    gain_lower_prod = int((2 ** g) * self.freq_upper_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain_var)
                        gain_lower_overflow_literals.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literal: ", overflow[0])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literals.append(gain_var)
                        print("ignored ", self.freq_lower_lin[omega], " in frequency = ", self.freqx_axis[omega])
                else:
                    gain_lower_prod = int(-1*(2 ** g) * self.freq_lower_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain_var)
                        gain_lower_overflow_literals.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                        print("overflow happened in the gain lower product: appended this to the sum coeff:", overflow[1], " with literal: ", overflow[0])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literals.append(gain_var)

            filter_upper_pb_coeffs = filter_coeffs + gain_freq_upper_prod_coeffs + filter_overflow_coeffs + gain_upper_overflow_coeffs
            filter_upper_pb_literals = filter_literals + gain_upper_literals + filter_overflow_literals + gain_upper_overflow_literals

            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literals):
                raise Exception("sumtin wong with upper filter pb")
            
            

            print("coeffs up: ",filter_upper_pb_coeffs)
            print("lit up: ",filter_upper_pb_literals)

            filter_lower_pb_coeffs = filter_coeffs + gain_freq_lower_prod_coeffs + filter_overflow_coeffs + gain_lower_overflow_coeffs
            filter_lower_pb_literals = filter_literals + gain_lower_literals + filter_overflow_literals + gain_lower_overflow_literals
            
            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")


            print("coeffs: ",filter_lower_pb_coeffs)
            print("lit: ",filter_lower_pb_literals)


            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with upper filter pb")
            
            cnf = PBEnc.atmost(lits=filter_upper_pb_literals, weights=filter_upper_pb_coeffs, bound=0, vpool=pool, encoding=encoding_type)
            cnf.extend(PBEnc.atleast(lits=filter_lower_pb_literals, weights=filter_lower_pb_coeffs, bound=0, vpool=pool, encoding=encoding_type).clauses)

            for clause in cnf.clauses:
                solver.add_clause(clause)

        # Bitshift SAT starts here

        # c0,w is always 0 except w=0
        for w in range(1, self.wordlength):
            solver.add_clause([-v2i(('c', 0, w))])

        solver.add_clause([v2i(('c', 0, 0))])

        # Input multiplexer
        for i in range(1, self.N + 1):
            alpha_lits = []
            beta_lits = []
            for a in range(i):
                for word in range(self.wordlength):
                    solver.add_clause([-v2i(('alpha', i, a)), -v2i(('c', a, word)), v2i(('l', i, word))])
                    solver.add_clause([-v2i(('alpha', i, a)), v2i(('c', a, word)), -v2i(('l', i, word))])
                    solver.add_clause([-v2i(('Beta', i, a)), -v2i(('c', a, word)), v2i(('r', i, word))])
                    solver.add_clause([-v2i(('Beta', i, a)), v2i(('c', a, word)), -v2i(('r', i, word))])

                alpha_lits.append(v2i(('alpha', i, a)))

                beta_lits.append(v2i(('Beta', i, a)))

            cnf = PBEnc.atmost(lits=alpha_lits, weights=[1]*len(alpha_lits), bound=1, vpool=pool, encoding=encoding_type)
            cnf.extend(PBEnc.atleast(lits=alpha_lits, weights=[1]*len(alpha_lits), bound=1, vpool=pool, encoding=encoding_type).clauses)

            for clause in cnf.clauses:
                solver.add_clause(clause)
            
            cnf = PBEnc.equals(lits=beta_lits, weights=[1]*len(beta_lits), bound=1, vpool=pool, encoding=encoding_type)

            for clause in cnf.clauses:
                solver.add_clause(clause)

        # Left Shifter
        for i in range(1, self.N + 1):
            gamma_lits = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('gamma', i, k)), -v2i(('l', i, j)), v2i(('s', i, j + k))])
                    solver.add_clause([-v2i(('gamma', i, k)), v2i(('l', i, j)), -v2i(('s', i, j + k))])

                gamma_lits.append(v2i(('gamma', i, k)))
            
            cnf = PBEnc.equals(lits=gamma_lits, weights=[1]*len(gamma_lits), bound=1, vpool=pool, encoding=encoding_type)

            for clause in cnf.clauses:
                solver.add_clause(clause)

            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('s', i, b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('l', i, self.wordlength - 1)), v2i(('l', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), v2i(('l', i, self.wordlength - 1)), -v2i(('l', i, self.wordlength - 2 - b))])

            solver.add_clause([-v2i(('l', i, self.wordlength - 1)), v2i(('s', i, self.wordlength - 1))])
            solver.add_clause([v2i(('l', i, self.wordlength - 1)), -v2i(('s', i, self.wordlength - 1))])

        for i in range(1, self.N + 1):
            for word in range(self.wordlength):
                solver.add_clause([-v2i(('delta', i)), -v2i(('s', i, word)), v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('s', i, word)), -v2i(('x', i, word))])
                solver.add_clause([-v2i(('delta', i)), -v2i(('r', i, word)), v2i(('u', i, word))])
                solver.add_clause([-v2i(('delta', i)), v2i(('r', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('s', i, word)), v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('s', i, word)), -v2i(('u', i, word))])
                solver.add_clause([v2i(('delta', i)), -v2i(('r', i, word)), v2i(('x', i, word))])
                solver.add_clause([v2i(('delta', i)), v2i(('r', i, word)), -v2i(('x', i, word))])

                solver.add_clause([v2i(('delta', i)), -v2i(('delta', i))])

        for i in range(1, self.N + 1):
            for word in range(self.wordlength):
                solver.add_clause([v2i(('u', i, word)), v2i(('epsilon', i)), -v2i(('y', i, word))])
                solver.add_clause([v2i(('u', i, word)), -v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), v2i(('epsilon', i)), v2i(('y', i, word))])
                solver.add_clause([-v2i(('u', i, word)), -v2i(('epsilon', i)), -v2i(('y', i, word))])

        for i in range(1, self.N + 1):
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

            for kf in range(1, self.wordlength):
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

            solver.add_clause([v2i(('epsilon', i)), v2i(('x', i, self.wordlength - 1)), v2i(('u', i, self.wordlength - 1)), -v2i(('z', i, self.wordlength - 1))])
            solver.add_clause([v2i(('epsilon', i)), -v2i(('x', i, self.wordlength - 1)), -v2i(('u', i, self.wordlength - 1)), v2i(('z', i, self.wordlength - 1))])
            solver.add_clause([-v2i(('epsilon', i)), v2i(('x', i, self.wordlength - 1)), -v2i(('u', i, self.wordlength - 1)), -v2i(('z', i, self.wordlength - 1))])
            solver.add_clause([-v2i(('epsilon', i)), -v2i(('x', i, self.wordlength - 1)), v2i(('u', i, self.wordlength - 1)), v2i(('z', i, self.wordlength - 1))])

        for i in range(1, self.N + 1):
            zeta_lits = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('zeta', i, k)), -v2i(('z', i, j + k)), v2i(('c', i, j))])
                    solver.add_clause([-v2i(('zeta', i, k)), v2i(('z', i, j + k)), -v2i(('c', i, j))])

                zeta_lits.append(v2i(('zeta', i, k)))
            
            cnf = PBEnc.equals(lits=zeta_lits, weights=[1]*len(zeta_lits), bound=1, vpool=pool, encoding=encoding_type)

            for clause in cnf.clauses:
                solver.add_clause(clause)

            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, self.wordlength - 1)), v2i(('c', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), v2i(('z', i, self.wordlength - 1)), -v2i(('c', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, b))])

            solver.add_clause([-v2i(('z', i, self.wordlength - 1)), v2i(('c', i, self.wordlength - 1))])
            solver.add_clause([v2i(('z', i, self.wordlength - 1)), -v2i(('c', i, self.wordlength - 1))])

            # Bound ci,0 to be odd number 
            solver.add_clause([v2i(('c', i, 0))])

        connected_coefficient = half_order + 1

        e_lits = []
        for m in range(half_order + 1):
            h_or_clause = []
            t_or_clauses = []

            for w in range(self.wordlength):
                h_or_clause.append(v2i(('h', m, w)))
            h_or_clause.append(v2i(('h0', m)))
            solver.add_clause(h_or_clause)

            for i in range(1, self.N + 1):
                for word in range(self.wordlength):
                    solver.add_clause([-v2i(('t', i, m)), -v2i(('e', m)), -v2i(('c', i, word)), v2i(('h', m, word))])
                    solver.add_clause([-v2i(('t', i, m)), -v2i(('e', m)), v2i(('c', i, word)), -v2i(('h', m, word))])

                t_or_clauses.append(v2i(('t', i, m)))
            solver.add_clause(t_or_clauses)

            e_lits.append(v2i(('e', m)))
        
        cnf = PBEnc.equals(lits=e_lits, weights=[1]*len(e_lits), bound=connected_coefficient, vpool=pool, encoding=encoding_type)

        for clause in cnf.clauses:
            solver.add_clause(clause)

        start_time = time.time()
        print("solver running")

        if solver.solve():
            print("solver sat")
            model = solver.get_model()
            print(model)
            end_time = time.time()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    var_index = v2i(('h', m, w)) - 1
                    bool_value = model[var_index] > 0  # Convert to boolean
                    print(f"h{m}_{w} = ", bool_value)
                    print(m, w, var_index + 1)

                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fh) * bool_value
                    elif w < self.fh:
                        fir_coef += 2 ** (-1 * (self.fh - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fh) * bool_value

                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ", self.h_res)

            gain_coef = 0
            for g in range(self.gain_wordlength):
                var_index = v2i(('gain', g))-1
                bool_value = model[var_index] > 0  # Convert to boolean
                print(f"gain{g}= ",v2i(('gain', g)) ,bool_value)

                if g < self.fg:
                    gain_coef += 2 ** -(self.fg - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fg) * bool_value

            self.gain_res = gain_coef
            print("gain Coeffs: ", self.gain_res)
        else:
            print("Unsatisfiable")
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
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]

        magnitude_response = np.abs(frequency_response)[:N//2]

        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        omega = frequencies * 2 * np.pi
        normalized_omega = omega / np.max(omega)
        self.ax1.set_ylim([-10, 10])

        freq_upper_lin_array = np.array(self.freq_upper_lin, dtype=np.float64)
        freq_lower_lin_array = np.array(self.freq_lower_lin, dtype=np.float64)

        self.freq_upper_lin = ((freq_upper_lin_array/((10**self.coef_accuracy)*(2**self.fh))) * self.gain_res).tolist()
        self.freq_lower_lin = ((freq_lower_lin_array/((10**self.coef_accuracy)*(2**self.fh))) * self.gain_res).tolist()

        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

        self.ax1.plot(normalized_omega, magnitude_response, color='y')

        if self.app:
            self.app.canvas.draw()

    def plot_validation(self):
        print("Validation plotter called")
        half_order = (self.order_current // 2)
        sf = SolverFunc(self.filter_type, self.order_current)
        computed_frequency_response = []
        
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            
            for j in range(half_order+1):
                cm_const = sf.cm_handler(j, omega)
                term_sum_exprs += self.h_res[j] * cm_const
            
            computed_frequency_response.append(np.abs(term_sum_exprs))

        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        self.ax2.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)

        if self.app:
            self.app.canvas.draw()

# Test inputs
filter_type = 0
order_upper = 10
accuracy = 1
adder_count = 20
wordlength = 6

# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
freq_upper = np.full(accuracy * order_upper, np.nan)
freq_lower = np.full(accuracy * order_upper, np.nan)

# Manually set specific values for the elements of freq_upper and freq_lower in dB
lower_half_point = int(0.4*(accuracy*order_upper))
upper_half_point = int(0.6*(accuracy*order_upper))
end_point = accuracy*order_upper

freq_upper[0:lower_half_point] = 10
freq_lower[0:lower_half_point] = -5

freq_upper[upper_half_point:end_point] = -5
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

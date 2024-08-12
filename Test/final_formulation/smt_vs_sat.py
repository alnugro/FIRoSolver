import numpy as np
from z3 import *
import time
import random

class SolverFunc1():
    def __init__(self,filter_type, order, accuracy):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.coef_accuracy = accuracy

    def db_to_linear(self,db_arr):
        nan_mask = np.isnan(db_arr)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return self.coef_accuracy
            cm=(2*np.cos(np.pi*omega*m))*self.coef_accuracy
            return int(cm)
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))
        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))
        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))

class SolverFunc2():
    def __init__(self,filter_type, order):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.overflow_count = 0

    def db_to_linear(self,db_arr):
        nan_mask = np.isnan(db_arr)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            cm=(2*np.cos(np.pi*omega*m))
            return cm
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))
        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))
        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))
        
    def overflow_handler(self, input_value, upper_bound, lower_bound, literal):
        self.overflow_count+=1
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
            print("somehting weird happens on overflow handler")
        return [overflow_lit, overflow_coef]
    

class FIRFilter1:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.app = app
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0
        self.coef_accuracy = 10**4
        self.ignore_lowerbound_lin = ignore_lowerbound_lin * self.coef_accuracy
        self.A_M = adder_count
        self.wordlength = wordlength
        self.order_current = int(self.order_upper)
        self.half_order = (self.order_current // 2)

    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        sf = SolverFunc1(self.filter_type, self.order_current, self.coef_accuracy)
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * self.coef_accuracy) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        h_int = [Int(f'h_int_{i}') for i in range(half_order + 1)]
        solver = Solver()
        solver.set(timeout=timeout)

        for i in range(len(self.freqx_axis)):
            term_sum_exprs = 0
            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue
            for j in range(half_order + 1):
                cm_const = sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs += h_int[j] * (2**(-1 * self.wordlength)) * cm_const
            solver.add(term_sum_exprs <= self.freq_upper_lin[i])
            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                solver.add(term_sum_exprs >= -self.freq_upper_lin[i])
                continue
            solver.add(term_sum_exprs >= self.freq_lower_lin[i])

        for i in range(half_order + 1):
            solver.add(h_int[i] <= 2**self.wordlength)
            solver.add(h_int[i] >= -1 * 2**self.wordlength)

        c_a = [Int(f'c_a{a}') for a in range(self.A_M + 1)]
        solver.add(c_a[0] == 1)
        c_sh_sg_a_i = [[Int(f'c_sh_sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(c_a[a + 1] == c_sh_sg_a_i[a][0] + c_sh_sg_a_i[a][1])
        c_a_i = [[Int(f'c_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        c_a_i_k = [[[Bool(f'c_a_i_k{a}{i}{k}') for k in range(self.A_M + 1)] for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            for i in range(2):
                for k in range(a + 1):
                    solver.add(Implies(c_a_i_k[a][i][k], c_a_i[a][i] == c_a[k]))
                solver.add(PbEq([(c_a_i_k[a][i][k], 1) for k in range(a + 1)], 1))
        c_sh_a_i = [[Int(f'c_sh_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        sh_a_i_s = [[[Bool(f'sh_a_i_s{a}{i}{s}') for s in range(2 * self.wordlength + 1)] for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            for i in range(2):
                for s in range(2 * self.wordlength + 1):
                    if s > self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][i][s] == False)
                    if s < self.wordlength and i == 0:
                        solver.add(sh_a_i_s[a][0][s] == sh_a_i_s[a][1][s])
                    shift = s - self.wordlength
                    solver.add(Implies(sh_a_i_s[a][i][s], c_sh_a_i[a][i] == (2 ** shift) * c_a_i[a][i]))
                solver.add(PbEq([(sh_a_i_s[a][i][s], 1) for s in range(2 * self.wordlength + 1)], 1))
        sg_a_i = [[Bool(f'sg_a_i{a}{i}') for i in range(2)] for a in range(self.A_M)]
        for a in range(self.A_M):
            solver.add(sg_a_i[a][0] + sg_a_i[a][1] <= 1)
            for i in range(2):
                solver.add(Implies(sg_a_i[a][i], -1 * c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
                solver.add(Implies(Not(sg_a_i[a][i]), c_sh_a_i[a][i] == c_sh_sg_a_i[a][i]))
        o_a_m_s_sg = [[[[Bool(f'o_a_m_s_sg{a}{i}{s}{sg}') for sg in range(2)] for s in range(2 * self.wordlength + 1)] for i in range(self.half_order + 1)] for a in range(self.A_M + 1)]
        for i in range(self.half_order + 1):
            for a in range(self.A_M + 1):
                for s in range(2 * self.wordlength + 1):
                    shift = s - self.wordlength
                    for sg in range(2):
                        solver.add(Implies(o_a_m_s_sg[a][i][s][sg], (-1 ** sg) * (2 ** shift) * c_a[a] == h_int[i]))
            solver.add(PbEq([(o_a_m_s_sg[a][i][s][sg], 1) for a in range(self.A_M + 1) for s in range(2 * self.wordlength + 1) for sg in range(2)], 1))

        start_time = time.time()
        result = solver.check()
        end_time = time.time()
        duration = end_time - start_time

        
        if result == sat:
            return duration, "sat"
        elif result == unsat:
            return duration, "unsat"
        else:
            return duration, "timeout"

class FIRFilter2:
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0
        self.wordlength = wordlength
        self.N = adder_count
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0
        self.coef_accuracy = 4
        self.ih = 2
        self.fh = self.wordlength - self.ih
        self.gain_wordlength = 6
        self.ig = 3
        self.fg = self.gain_wordlength - self.ig
        self.gain_upperbound = 1
        self.gain_lowerbound = 1
        self.gain_bound_accuracy = 2
        self.ignore_lowerbound_lin = ignore_lowerbound_lin * (10**self.coef_accuracy) * (2**self.fh)

    def runsolver(self):
        self.order_current = int(self.order_upper)
        half_order = (self.order_current // 2)
        sf = SolverFunc2(self.filter_type, self.order_current)
        self.freq_upper_lin = [int((sf.db_to_linear(f)) * (10**self.coef_accuracy) * (2**self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [int((sf.db_to_linear(f)) * (10**self.coef_accuracy) * (2**self.fh)) if not np.isnan(sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        h = [[Bool(f'h{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = [Bool(f'gain{g}') for g in range(self.gain_wordlength)]
        solver = Solver()
        solver.set(timeout=timeout)

        gain_coeffs = []
        gain_literalls = []
        self.gain_upperbound_int = int(self.gain_upperbound * 2**self.fg * (10**self.gain_bound_accuracy))
        self.gain_lowerbound_int = int(self.gain_lowerbound * 2**self.fg * (10**self.gain_bound_accuracy))
        for g in range(self.gain_wordlength):
            gain_coeffs.append((2**g) * (10**self.gain_bound_accuracy))
            gain_literalls.append(gain[g])
        pb_gain_pairs = [(gain_literalls[i], gain_coeffs[i]) for i in range(len(gain_literalls))]
        solver.add(PbLe(pb_gain_pairs, self.gain_upperbound_int))
        solver.add(PbGe(pb_gain_pairs, self.gain_lowerbound_int))

        filter_literals = []
        filter_coeffs = []
        gain_freq_upper_prod_coeffs = []
        gain_freq_lower_prod_coeffs = []
        filter_upper_pb_pairs = []
        filter_lower_pb_pairs = []
        filter_overflow_literalls = []
        filter_overflow_coeffs = []
        gain_upper_overflow_literalls = []
        gain_upper_overflow_coeffs = []
        gain_lower_overflow_literalls = []
        gain_lower_overflow_coeffs = []
        gain_upper_literalls = []
        gain_lower_literalls = []
        max_positive_int_pbfunc = 2147483647
        max_negative_int_pbfunc = -2147483648

        for omega in range(len(self.freqx_axis)):
            if np.isnan(self.freq_lower_lin[omega]):
                continue
            gain_literalls.clear()
            filter_literals.clear()
            filter_coeffs.clear()
            gain_freq_upper_prod_coeffs.clear()
            gain_freq_lower_prod_coeffs.clear()
            filter_upper_pb_pairs.clear()
            filter_lower_pb_pairs.clear()
            filter_overflow_literalls.clear()
            filter_overflow_coeffs.clear()
            gain_upper_overflow_literalls.clear()
            gain_upper_overflow_coeffs.clear()
            gain_lower_overflow_literalls.clear()
            gain_lower_overflow_coeffs.clear()
            gain_upper_literalls.clear()
            gain_lower_literalls.clear()
            for m in range(half_order + 1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w == self.wordlength - 1:
                        cm_word_prod = int(cm * (10**self.coef_accuracy) * (-1 * (2**w)) * (2**self.fg))
                    else:
                        cm_word_prod = int(cm * (10**self.coef_accuracy) * (2**w) * 2**self.fg)
                    if cm_word_prod > max_positive_int_pbfunc or cm_word_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(cm_word_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, h[m][w])
                        filter_overflow_literalls.extend(overflow[0])
                        filter_overflow_coeffs.extend(overflow[1])
                    else:
                        filter_coeffs.append(cm_word_prod)
                        filter_literals.append(h[m][w])
            for g in range(self.gain_wordlength):
                gain_upper_prod = int(-1 * (2**g) * self.freq_upper_lin[omega])
                if gain_upper_prod > max_positive_int_pbfunc or gain_upper_prod < max_negative_int_pbfunc:
                    overflow = sf.overflow_handler(gain_upper_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain[g])
                    gain_upper_overflow_literalls.extend(overflow[0])
                    gain_upper_overflow_coeffs.extend(overflow[1])
                else:
                    gain_freq_upper_prod_coeffs.append(gain_upper_prod)
                    gain_upper_literalls.append(gain[g])
                if self.freq_lower_lin[omega] < self.ignore_lowerbound_lin:
                    gain_lower_prod = int((2**g) * self.freq_upper_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literalls.append(gain[g])
                else:
                    gain_lower_prod = int(-1 * (2**g) * self.freq_lower_lin[omega])
                    if gain_lower_prod > max_positive_int_pbfunc or gain_lower_prod < max_negative_int_pbfunc:
                        overflow = sf.overflow_handler(gain_lower_prod, max_positive_int_pbfunc, max_negative_int_pbfunc, gain[g])
                        gain_lower_overflow_literalls.extend(overflow[0])
                        gain_lower_overflow_coeffs.extend(overflow[1])
                    else:
                        gain_freq_lower_prod_coeffs.append(gain_lower_prod)
                        gain_lower_literalls.append(gain[g])

            filter_upper_pb_coeffs = filter_coeffs + gain_freq_upper_prod_coeffs + filter_overflow_coeffs + gain_upper_overflow_coeffs
            filter_upper_pb_literalls = filter_literals + gain_upper_literalls + filter_overflow_literalls + gain_upper_overflow_literalls
            if len(filter_upper_pb_coeffs) != len(filter_upper_pb_literalls):
                raise Exception("Something wrong with upper filter pb")
            filter_upper_pb_pairs = [(filter_upper_pb_literalls[i], filter_upper_pb_coeffs[i]) for i in range(len(filter_upper_pb_literalls))]
            solver.add(PbLe(filter_upper_pb_pairs, 0))

            filter_lower_pb_coeffs = filter_coeffs + gain_freq_lower_prod_coeffs + filter_overflow_coeffs + gain_lower_overflow_coeffs
            filter_lower_pb_literalls = filter_literals + gain_lower_literalls + filter_overflow_literalls + gain_lower_overflow_literalls
            if len(filter_lower_pb_coeffs) != len(filter_lower_pb_literalls):
                raise Exception("Something wrong with lower filter pb")
            filter_lower_pb_pairs = [(filter_lower_pb_literalls[i], filter_lower_pb_coeffs[i]) for i in range(len(filter_lower_pb_literalls))]
            solver.add(PbGe(filter_lower_pb_pairs, 0))

        c = [[Bool(f'c{i}{w}') for w in range(self.wordlength)] for i in range(self.N + 1)]
        l = [[Bool(f'l{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        r = [[Bool(f'r{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        alpha = [[Bool(f'alpha{i}{a}') for a in range(i)] for i in range(1, self.N + 1)]
        beta = [[Bool(f'beta{i}{a}') for a in range(i)] for i in range(1, self.N + 1)]
        for w in range(1, self.wordlength):
            solver.add(Not(c[0][w]))
        solver.add(c[0][0])
        for i in range(1, self.N + 1):
            alpha_sum = []
            beta_sum = []
            for a in range(i):
                for word in range(self.wordlength):
                    clause1_1 = Or(Not(alpha[i - 1][a]), Not(c[a][word]), l[i - 1][word])
                    clause1_2 = Or(Not(alpha[i - 1][a]), c[a][word], Not(l[i - 1][word]))
                    solver.add(And(clause1_1, clause1_2))
                    clause2_1 = Or(Not(beta[i - 1][a]), Not(c[a][word]), r[i - 1][word])
                    clause2_2 = Or(Not(beta[i - 1][a]), c[a][word], Not(r[i - 1][word]))
                    solver.add(And(clause2_1, clause2_2))
                alpha_sum.append((alpha[i - 1][a], 1))
                beta_sum.append((beta[i - 1][a], 1))
            solver.add(PbEq(alpha_sum, 1))
            solver.add(PbEq(beta_sum, 1))

        gamma = [[Bool(f'gamma{i}{k}') for k in range(self.wordlength - 1)] for i in range(1, self.N + 1)]
        s = [[Bool(f's{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        for i in range(1, self.N + 1):
            gamma_sum = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    clause3_1 = Or(Not(gamma[i - 1][k]), Not(l[i - 1][j]), s[i - 1][j + k])
                    clause3_2 = Or(Not(gamma[i - 1][k]), l[i - 1][j], Not(s[i - 1][j + k]))
                    solver.add(And(clause3_1, clause3_2))
                gamma_sum.append((gamma[i - 1][k], 1))
            solver.add(PbEq(gamma_sum, 1))
            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    clause4 = Or(Not(gamma[i - 1][kf]), Not(s[i - 1][b]))
                    clause5 = Or(Not(gamma[i - 1][kf]), Not(l[i - 1][self.wordlength - 1]), l[i - 1][self.wordlength - 2 - b])
                    clause6 = Or(Not(gamma[i - 1][kf]), l[i - 1][self.wordlength - 1], Not(l[i - 1][self.wordlength - 2 - b]))
                    solver.add(clause4)
                    solver.add(clause5)
                    solver.add(clause6)
            clause7_1 = Or(Not(l[i - 1][self.wordlength - 1]), s[i - 1][self.wordlength - 1])
            clause7_2 = Or(l[i - 1][self.wordlength - 1], Not(s[i - 1][self.wordlength - 1]))
            solver.add(And(clause7_1, clause7_2))

        delta = [Bool(f'delta{i}') for i in range(1, self.N + 1)]
        u = [[Bool(f'u{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        x = [[Bool(f'x{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        for i in range(1, self.N + 1):
            for word in range(self.wordlength):
                clause8_1 = Or(Not(delta[i - 1]), Not(s[i - 1][word]), x[i - 1][word])
                clause8_2 = Or(Not(delta[i - 1]), s[i - 1][word], Not(x[i - 1][word]))
                solver.add(And(clause8_1, clause8_2))
                clause9_1 = Or(Not(delta[i - 1]), Not(r[i - 1][word]), u[i - 1][word])
                clause9_2 = Or(Not(delta[i - 1]), r[i - 1][word], Not(u[i - 1][word]))
                solver.add(And(clause9_1, clause9_2))
                clause10_1 = Or(delta[i - 1], Not(s[i - 1][word]), u[i - 1][word])
                clause10_2 = Or(delta[i - 1], s[i - 1][word], Not(u[i - 1][word]))
                solver.add(And(clause10_1, clause10_2))
                clause11_1 = Or(delta[i - 1], Not(r[i - 1][word]), x[i - 1][word])
                clause11_2 = Or(delta[i - 1], r[i - 1][word], Not(x[i - 1][word]))
                solver.add(And(clause11_1, clause11_2))
                solver.add(Or(delta[i - 1], Not(delta[i - 1])))

        epsilon = [Bool(f'epsilon{i}') for i in range(1, self.N + 1)]
        y = [[Bool(f'y{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        for i in range(1, self.N + 1):
            for word in range(self.wordlength):
                clause12 = Or(u[i - 1][word], epsilon[i - 1], Not(y[i - 1][word]))
                clause13 = Or(u[i - 1][word], Not(epsilon[i - 1]), y[i - 1][word])
                clause14 = Or(Not(u[i - 1][word]), epsilon[i - 1], y[i - 1][word])
                clause15 = Or(Not(u[i - 1][word]), Not(epsilon[i - 1]), Not(y[i - 1][word]))
                solver.add(clause12)
                solver.add(clause13)
                solver.add(clause14)
                solver.add(clause15)

        z = [[Bool(f'z{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        cout = [[Bool(f'cout{i}{w}') for w in range(self.wordlength)] for i in range(1, self.N + 1)]
        for i in range(1, self.N + 1):
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
            for kf in range(1, self.wordlength):
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
            clause44 = Or(epsilon[i - 1], x[i - 1][self.wordlength - 1], u[i - 1][self.wordlength - 1], Not(z[i - 1][self.wordlength - 1]))
            clause45 = Or(epsilon[i - 1], Not(x[i - 1][self.wordlength - 1]), Not(u[i - 1][self.wordlength - 1]), z[i - 1][self.wordlength - 1])
            clause46 = Or(Not(epsilon[i - 1]), x[i - 1][self.wordlength - 1], Not(u[i - 1][self.wordlength - 1]), Not(z[i - 1][self.wordlength - 1]))
            clause47 = Or(Not(epsilon[i - 1]), Not(x[i - 1][self.wordlength - 1]), u[i - 1][self.wordlength - 1], z[i - 1][self.wordlength - 1])
            solver.add(clause44)
            solver.add(clause45)
            solver.add(clause46)
            solver.add(clause47)

        zeta = [[Bool(f'zeta{i}{k}') for k in range(self.wordlength - 1)] for i in range(1, self.N + 1)]
        for i in range(1, self.N + 1):
            zeta_sum = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    clause48_1 = Or(Not(zeta[i - 1][k]), Not(z[i - 1][j + k]), c[i][j])
                    clause48_2 = Or(Not(zeta[i - 1][k]), z[i - 1][j + k], Not(c[i][j]))
                    solver.add(And(clause48_1, clause48_2))
                zeta_sum.append((zeta[i - 1][k], 1))
            solver.add(PbEq(zeta_sum, 1))
            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    clause49_1 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][self.wordlength - 1]), c[i][self.wordlength - 2 - b])
                    clause49_2 = Or(Not(zeta[i - 1][kf]), z[i - 1][self.wordlength - 1], Not(c[i][self.wordlength - 2 - b]))
                    solver.add(clause49_1)
                    solver.add(clause49_2)
                    clause50 = Or(Not(zeta[i - 1][kf]), Not(z[i - 1][b]))
                    solver.add(clause50)
            clause51_1 = Or(Not(z[i - 1][self.wordlength - 1]), c[i][self.wordlength - 1])
            clause51_2 = Or(z[i - 1][self.wordlength - 1], Not(c[i][self.wordlength - 1]))
            solver.add(And(clause51_1, clause51_2))
        solver.add(c[0][0])

        connected_coefficient = half_order + 1
        h = [[Bool(f'h{m}_{w}') for w in range(self.wordlength)] for m in range(half_order + 1)]
        h0 = [Bool(f'h0{m}') for m in range(half_order + 1)]
        t = [[Bool(f't{i}_{m}') for m in range(half_order + 1)] for i in range(1, self.N + 1)]
        e = [Bool(f'e{m}') for m in range(half_order + 1)]
        e_sum = []
        for m in range(half_order + 1):
            h_or_clause = []
            t_or_clauses = []
            for w in range(self.wordlength):
                h_or_clause.append(h[m][w])
            h_or_clause.append(h0[m])
            solver.add(Or(h_or_clause))
            for i in range(1, self.N + 1):
                for word in range(self.wordlength):
                    clause52_1 = Or(Not(t[i - 1][m]), Not(e[m]), Not(c[i][word]), h[m][word])
                    clause52_2 = Or(Not(t[i - 1][m]), Not(e[m]), c[i][word], Not(h[m][word]))
                    solver.add(And(clause52_1, clause52_2))
                t_or_clauses.append(t[i - 1][m])
            solver.add(Or(t_or_clauses))
            e_sum.append((e[m], 1))
        solver.add(PbEq(e_sum, connected_coefficient))

        start_time = time.time()
        result = solver.check()
        end_time = time.time()
        duration = end_time - start_time

        
        if result == sat:
            return duration, "sat"
        elif result == unsat:
            return duration, "unsat"
        else:
            return duration, "timeout"


# Initialize global variable
it = 4
timeout = 300000  # 10 minutes in milliseconds


def generate_random_filter_params():
    global it
    filter_type = 0
    order_upper = it
    accuracy = random.choice([1, 2, 3, 4, 5])
    adder_count = np.abs(it - (random.choice([1, 2, 3, 4, it - 4])))
    wordlength = random.choice([10, 12, 14])
    upper_cutoff = random.choice([0.6, 0.7, 0.8, 0.9])
    lower_cutoff = random.choice([0.2, 0.3, 0.4, 0.5])
    lower_half_point = int(lower_cutoff * (accuracy * order_upper))
    upper_half_point = int(upper_cutoff * (accuracy * order_upper))
    end_point = accuracy * order_upper
    freqx_axis = np.linspace(0, 1, accuracy * order_upper)
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)
    passband_upperbound = random.choice([0, 1, 2, 3, 4, 5])
    passband_lowerbound = random.choice([0, -1, -2])
    stopband_upperbound = random.choice([-10,-20,-30, -40, -50])
    stopband_lowerbound = -1000
    freq_upper[0:lower_half_point] = passband_upperbound
    freq_lower[0:lower_half_point] = passband_lowerbound
    freq_upper[upper_half_point:end_point] = stopband_upperbound
    freq_lower[upper_half_point:end_point] = stopband_lowerbound
    ignore_lowerbound_lin = 0.0001
    it += 2
    return (filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound)


# Write header
with open("z3_smt_vs_sat.txt", "w") as file:
    file.write("time_smt, result_smt, time_sat, result_sat, filter_type, order_upper, accuracy, adder_count, wordlength, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound\n")

results = []
for i in range(50):
    print("running test: ", i)
    params = generate_random_filter_params()
    filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound = params
    fir_filter1 = FIRFilter1(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)
    fir_filter2 = FIRFilter2(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)
    time1, result1 = fir_filter1.runsolver()
    time2, result2 = fir_filter2.runsolver()
    results.append((time1, result1, time2, result2, *params))
    with open("z3_smt_vs_sat.txt", "a") as file:
        file.write(f"{time1}, {result1}, {time2}, {result2}, {filter_type}, {order_upper}, {accuracy}, {adder_count}, {wordlength}, {upper_cutoff}, {lower_cutoff}, {passband_upperbound}, {passband_lowerbound}, {stopband_upperbound}, {stopband_lowerbound}\n")
    print("test ", i, " is completed")

print("Benchmark completed and results saved to z3_smt_vs_sat.txt")


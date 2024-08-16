import numpy as np
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import time
from sat_variable_handler import VariableMapper
from pb2cnf import PB2CNF
from rat2bool import Rat2bool
import multiprocessing


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
    def __init__(self, filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength, app=None):
        self.filter_type = filter_type
        self.order_upper = order_upper
        self.freqx_axis = freqx_axis
        self.freq_upper = freq_upper
        self.freq_lower = freq_lower
        self.h_res = []
        self.gain_res = 0
        self.model = None

    
        self.N = adder_count

        self.app = app
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.freq_upper_lin = 0
        self.freq_lower_lin = 0

        self.wordlength = wordlength
        self.intW = 4
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound = 1.4
        self.gain_lowerbound = 1

        self.ignore_lowerbound = ignore_lowerbound

        self.order_current = int(self.order_upper)
        self.sf = SolverFunc(self.filter_type, self.order_current)
        self.freq_upper_lin = [self.sf.db_to_linear(f) if not np.isnan(self.sf.db_to_linear(f)) else np.nan for f in self.freq_upper]
        self.freq_lower_lin = [self.sf.db_to_linear(f) if not np.isnan(self.sf.db_to_linear(f)) else np.nan for f in self.freq_lower]
        self.ignore_lowerbound_np = np.array(self.ignore_lowerbound, dtype=float)
        self.ignore_lowerbound = self.sf.db_to_linear(self.ignore_lowerbound_np) 
       

    def runsolver_internal(self):
        half_order = (self.order_current // 2)
        
        print("solver called")
        var_mapper = VariableMapper(half_order, self.wordlength, self.N)

        def v2i(var_tuple):
            return var_mapper.tuple_to_int(var_tuple)

        def i2v(var_int):
            return var_mapper.int_to_var_name(var_int)
        
        #initiate top var
        top_var = var_mapper.max_int_value
        pb2cnf = PB2CNF(top_var)
        r2b = Rat2bool()

        print("before ignore lower than:", self.ignore_lowerbound)
        
        print("filter order:", self.order_current)
        print("ignore lower than:", self.ignore_lowerbound)

        solver = Solver(name='cadical195')



        #bound the gain to upper and lowerbound
        gain_literals = []

        for g in range(self.wordlength):
            gain_literals.append(v2i(('gain', g)))
            # print("gain lits :", v2i(('gain', g)))

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
        print(gain_literals)
        print(self.fracW)

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
                cm = self.sf.cm_handler(m, self.freqx_axis[omega])
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
                print("ignored ", self.freq_lower_lin[omega], " in frequency = ", self.freqx_axis[omega])
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
            print("\nbefore weight low: ",filter_lower_pb_weights)

            filter_lower_pb_weights = r2b.frac2round(filter_lower_pb_weights,self.wordlength,self.fracW)

            filter_lower_pb_literals = filter_literals + gain_lower_literals


            print("weight low: ",filter_lower_pb_weights)
            print("lit low: ",filter_lower_pb_literals)
            
            if len(filter_lower_pb_weights) != len(filter_lower_pb_literals):
                raise Exception("sumtin wong with lower filter pb")
            
            cnf4 = pb2cnf.atleast(weight=filter_lower_pb_weights,lits=filter_lower_pb_literals,bounds=0,fracW=self.fracW)

            for clause in cnf4:
                solver.add_clause(clause)
            
        

        # Bitshift SAT starts here

        # c0,w is always 0 except 1
        for w in range(self.fracW+1, self.wordlength):
            solver.add_clause([-v2i(('c', 0, w))])

        for w in range(self.fracW):
            solver.add_clause([-v2i(('c', 0, w))])

        solver.add_clause([v2i(('c', 0, self.fracW))])

        alpha_lits = []
        beta_lits = []

        # Input multiplexer
        for i in range(1, self.N + 1):
            alpha_lits.clear()
            beta_lits.clear()
            for a in range(i):
                for word in range(self.wordlength):
                    solver.add_clause([-v2i(('alpha', i, a)), -v2i(('c', a, word)), v2i(('l', i, word))])
                    solver.add_clause([-v2i(('alpha', i, a)), v2i(('c', a, word)), -v2i(('l', i, word))])
                    solver.add_clause([-v2i(('Beta', i, a)), -v2i(('c', a, word)), v2i(('r', i, word))])
                    solver.add_clause([-v2i(('Beta', i, a)), v2i(('c', a, word)), -v2i(('r', i, word))])

                alpha_lits.append(v2i(('alpha', i, a)))
                beta_lits.append(v2i(('Beta', i, a)))

            cnf5 = pb2cnf.equal_card_one(alpha_lits)
            for clause in cnf5:
                solver.add_clause(clause)

            cnf6 = pb2cnf.equal_card_one(beta_lits)
            for clause in cnf6:
                solver.add_clause(clause)

        gamma_lits = []
        # Left Shifter
        for i in range(1, self.N + 1):
            gamma_lits.clear()
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('gamma', i, k)), -v2i(('l', i, j)), v2i(('s', i, j + k))])
                    solver.add_clause([-v2i(('gamma', i, k)), v2i(('l', i, j)), -v2i(('s', i, j + k))])

                gamma_lits.append(v2i(('gamma', i, k)))
            
            cnf7 = pb2cnf.equal_card_one(gamma_lits)
            for clauses in cnf7:
                solver.add_clause(clauses)

            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('s', i, b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('l', i, self.wordlength - 1)), v2i(('l', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), v2i(('l', i, self.wordlength - 1)), -v2i(('l', i, self.wordlength - 2 - b))])

            solver.add_clause([-v2i(('l', i, self.wordlength - 1)), v2i(('s', i, self.wordlength - 1))])
            solver.add_clause([v2i(('l', i, self.wordlength - 1)), -v2i(('s', i, self.wordlength - 1))])
        
            
        #delta selector
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
        
        zeta_lits = []
        for i in range(1, self.N + 1):
            zeta_lits.clear()
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('zeta', i, k)), -v2i(('z', i, j + k)), v2i(('c', i, j))])
                    solver.add_clause([-v2i(('zeta', i, k)), v2i(('z', i, j + k)), -v2i(('c', i, j))])

                zeta_lits.append(v2i(('zeta', i, k)))

            cnf8 = pb2cnf.equal_card_one(zeta_lits)

            for clauses in cnf8:
                solver.add_clause(clauses)

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
        
        # cnf9 = pb2cnf.equal([1], [e_lits],connected_coefficient,0)
        for el in e_lits:
            solver.add_clause([el])
        

        start_time = time.time()
        print("solver running")

        satifiability = 'unsat'

        if solver.solve():
            satifiability = 'sat'
            print("solver sat")
            self.model = solver.get_model()
            # print(model)
            end_time = time.time()

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    var_index = v2i(('h', m, w)) - 1
                    bool_value = self.model[var_index] > 0  # Convert to boolean
                    # print(f"h{m}_{w} = ", bool_value)
                    # print(m, w, var_index + 1)

                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * bool_value
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fracW) * bool_value

                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ", self.h_res)

            gain_coef = 0
            for g in range(self.wordlength):
                var_index = v2i(('gain', g))-1
                bool_value = self.model[var_index] > 0  # Convert to boolean
                # print(f"gain{g}= ",v2i(('gain', g)) ,bool_value)

                if g < self.fracW:
                    gain_coef += 2 ** -(self.fracW - g) * bool_value
                else:
                    gain_coef += 2 ** (g - self.fracW) * bool_value

            self.gain_res = gain_coef
            print("gain Coeffs: ", self.gain_res)
            self.print_model_results(var_mapper, self.model)
        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")
        solver.delete()

        return duration, satifiability
    
    def print_model_results(self, var_mapper, model):
        """
        This function prints out all the variables in var_mapper along with their truth values
        based on the model provided by the SAT solver.
        
        :param var_mapper: An instance of the VariableMapper class.
        :param model: A list representing the model obtained from the SAT solver.
        """
        for var_int, var_name in var_mapper.int_to_var.items():
            # Convert the variable integer index to the model's 0-based index
            # A positive value in the model means True, negative means False.
            value = model[var_int - 1] > 0  # model is 0-based, var_int is 1-based
            print(f'{var_name}: {value}')


    def plot_result(self, result_coef):
        if not self.h_res:
            return
        print("result plotter called")
        print(f"gain res in plotter: {self.gain_res}")
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

        self.freq_upper_lin = (freq_upper_lin_array*self.gain_res).tolist()
        self.freq_lower_lin = (freq_lower_lin_array*self.gain_res).tolist()

        self.ax1.scatter(self.freqx_axis, self.freq_upper_lin, color='r', s=20, picker=5)
        self.ax1.scatter(self.freqx_axis, self.freq_lower_lin, color='b', s=20, picker=5)

        self.ax1.plot(normalized_omega, magnitude_response, color='y')

        if self.app:
            self.app.canvas.draw()

    def plot_validation(self):
        if not self.h_res:
            return

        print("Validation plotter called")
        half_order = (self.order_current // 2)
        computed_frequency_response = []
        
        for i in range(len(self.freqx_axis)):
            omega = self.freqx_axis[i]
            term_sum_exprs = 0
            
            for j in range(half_order+1):
                cm_const = self.sf.cm_handler(j, omega)
                term_sum_exprs += self.h_res[j] * cm_const
            
            computed_frequency_response.append(np.abs(term_sum_exprs))

        self.ax1.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')
        self.ax2.plot([x/1 for x in self.freqx_axis], computed_frequency_response, color='green', label='Computed Frequency Response')

        self.ax2.set_ylim(-10,10)

        if self.app:
            self.app.canvas.draw()

    def runsolver(self, timeout=0):
        if timeout == 0:
            return self.runsolver_internal()

        #added function to do timeout in PySat

        # Use a manager to create shared data structures
        manager = multiprocessing.Manager()
        shared_h_res = manager.list()  # Shared list for h_res
        shared_gain_res = manager.list()  # Shared list for gain_res
        shared_result = manager.dict()  # Shared dictionary to store results

        # Set up multiprocessing
        process = multiprocessing.Process(target=run_solver_wrapper, args=(shared_h_res, shared_gain_res, shared_result, self))
        process.start()

        # Wait for the process to finish or timeout
        process.join(timeout)

        if process.is_alive():
            print("Solver timed out! Terminating...")
            process.terminate()
            process.join()  # Ensure the parent wait for the child
            return timeout, 'timeout'

        # If process finished within the timeout, return the results
        self.h_res = list(shared_h_res)  
        self.gain_res = list(shared_gain_res) 
        print(f"gain res in process: {shared_gain_res}")

        duration = shared_result.get('duration', None)
        satisfiability = shared_result.get('satisfiability', 'unsat')
        self.model = shared_result.get('model', None)
        return duration, satisfiability

def run_solver_wrapper(shared_h_res, shared_gain_res, shared_result, fir_filter):
    """Wrapper function to call `runsolver_internal` and save the result."""
    duration, satisfiability = fir_filter.runsolver_internal()

    # Copy the results into the shared list to ensure they are updated
    shared_h_res[:] = fir_filter.h_res
    shared_gain_res[:] = [fir_filter.gain_res] 

    shared_result['duration'] = duration
    shared_result['satisfiability'] = satisfiability
    shared_result['model'] = fir_filter.model if satisfiability == 'sat' else None

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_upper = 20
    accuracy = 5
    adder_count = 8
    wordlength = 15

    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, accuracy*order_upper) #according to Mr. Kumms paper
    freq_upper = np.full(accuracy * order_upper, np.nan)
    freq_lower = np.full(accuracy * order_upper, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3*(accuracy*order_upper))
    upper_half_point = int(0.9*(accuracy*order_upper))
    end_point = accuracy*order_upper

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = -1

    freq_upper[upper_half_point:end_point] = -40
    freq_lower[upper_half_point:end_point] = -1000




    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    # Create FIRFilter instance
    fir_filter = FIRFilterPysat(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound, adder_count, wordlength)

    # Run solver and plot result
    fir_filter.runsolver()
    fir_filter.plot_result(fir_filter.h_res)
    fir_filter.plot_validation()

    # Show plot
    plt.show()

    # 0.12361645698547363, unsat, 0.1798112392425537, sat, 0.48868274688720703, sat, 0, 6, 5, 2, 10, 0.9, 0.3, 5, -1, -40, -1000

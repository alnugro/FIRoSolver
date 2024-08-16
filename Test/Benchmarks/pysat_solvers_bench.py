import numpy as np
from pysat.solvers import Solver
import matplotlib.pyplot as plt
import time
from sat_variable_handler import VariableMapper
from pb2cnf import PB2CNF
from rat2bool import Rat2bool
import random
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError

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
       

    def runsolver_internal(self, solver_option):
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

        solver = Solver(name=solver_option)



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


        cnf5 = pb2cnf.equal_card_one(alpha_lits)
        for clause in cnf5:
            solver.add_clause(clause)

        cnf6 = pb2cnf.equal_card_one(beta_lits)

        for clause in cnf6:
            solver.add_clause(clause)

        gamma_lits = []
        # Left Shifter
        for i in range(1, self.N + 1):
            gamma_lits = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('gamma', i, k)), -v2i(('l', i, j)), v2i(('s', i, j + k))])
                    solver.add_clause([-v2i(('gamma', i, k)), v2i(('l', i, j)), -v2i(('s', i, j + k))])

                gamma_lits.append(v2i(('gamma', i, k)))

            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('s', i, b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), -v2i(('l', i, self.wordlength - 1)), v2i(('l', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('gamma', i, kf)), v2i(('l', i, self.wordlength - 1)), -v2i(('l', i, self.wordlength - 2 - b))])

            solver.add_clause([-v2i(('l', i, self.wordlength - 1)), v2i(('s', i, self.wordlength - 1))])
            solver.add_clause([v2i(('l', i, self.wordlength - 1)), -v2i(('s', i, self.wordlength - 1))])
        
        cnf7 = pb2cnf.equal_card_one(gamma_lits)
        for clauses in cnf7:
            solver.add_clause(clauses)

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
        
        zeta_lits = []
        for i in range(1, self.N + 1):
            zeta_lits = []
            for k in range(self.wordlength - 1):
                for j in range(self.wordlength - 1 - k):
                    solver.add_clause([-v2i(('zeta', i, k)), -v2i(('z', i, j + k)), v2i(('c', i, j))])
                    solver.add_clause([-v2i(('zeta', i, k)), v2i(('z', i, j + k)), -v2i(('c', i, j))])

                zeta_lits.append(v2i(('zeta', i, k)))

            for kf in range(1, self.wordlength - 1):
                for b in range(kf):
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, self.wordlength - 1)), v2i(('c', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), v2i(('z', i, self.wordlength - 1)), -v2i(('c', i, self.wordlength - 2 - b))])
                    solver.add_clause([-v2i(('zeta', i, kf)), -v2i(('z', i, b))])

            solver.add_clause([-v2i(('z', i, self.wordlength - 1)), v2i(('c', i, self.wordlength - 1))])
            solver.add_clause([v2i(('z', i, self.wordlength - 1)), -v2i(('c', i, self.wordlength - 1))])

            # Bound ci,0 to be odd number 
            solver.add_clause([v2i(('c', i, 0))])

        cnf8 = pb2cnf.equal_card_one(zeta_lits)

        for clauses in cnf8:
            solver.add_clause(clauses)

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
        
        cnf9 = pb2cnf.equal([1], [e_lits],connected_coefficient,0)
        for clauses in cnf9:
            solver.add_clause(clauses)
        

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
        else:
            print("Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")
        

        return f"{duration},{satifiability}"


    def runsolver(self, timeout):
        solver_dict = {
            'cadical103': None,
            'cadical153': None,
            'cadical195': None,
            'cryptominisat5': None,
            'gluecard30': None,
            'gluecard41': None,
            'glucose30': None,
            'glucose41': None,
            'glucose421': None,
            'lingeling': None,
            'maplechrono': None,
            'maplecm': None,
            'maplesat': None,
            'mergesat30': None,
            'minicard': None,
            'minisat22': None,
            'minisat-gh': None
        }
        self.num_instances = len(solver_dict)
        
        with ProcessPool(max_workers=self.num_instances) as pool:
            futures = {}
            for solver in solver_dict.keys():
                future = pool.schedule(self.run_solver_with_timeout, args=[solver, timeout], timeout=timeout)
                futures[future] = solver

            for future in futures:
                solver = futures[future]
                try:
                    result = future.result()  # Will wait for the result or the timeout
                    solver_dict[solver] = result
                except TimeoutError:
                    solver_dict[solver] = f"{timeout},Timeout"
                except ProcessExpired as error:
                    solver_dict[solver] = f"{timeout},Expired: {error.exitcode}"
                except CancelledError:
                    solver_dict[solver] = f"{timeout},Cancelled"
                except Exception as error:
                    solver_dict[solver] = f"{timeout},Error: {str(error)}"

        # Build the final result string
        end_result = ""
        for solver in solver_dict:
            end_result += f"{solver_dict[solver]},"
        return end_result

    def run_solver_with_timeout(self, solver, timeout):
        """Wrapper function to call `runsolver_internal` and handle timeouts."""
        result = self.runsolver_internal(solver)

        return f"{result}"

# Initialize global variable
it = 1
timeout = 3600
random_seed = 1
random.seed(random_seed)


def generate_random_filter_params():
    global it
    iter = int(it)
    filter_type = 0
    order_upper = iter
    accuracy = random.choice([1, 2, 3, 4, 5])
    adder_count = np.abs(iter - (random.choice([1, 2, 3, 4, iter - 4])))
    wordlength = random.choice([10, 12, 14, 16])
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
    ignore_lowerbound_lin = -10
    it += 0.1
    return (filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound)


if __name__ == '__main__':

    # Write header
    with open("pysat_solver_bench.txt", "w") as file:
        file.write("cadical103_time, cadical103_result, cadical153_time, cadical153_result, cadical195_time, cadical195_result, cryptominisat5_time, cryptominisat5_result, gluecard30_time, gluecard30_result, gluecard41_time, gluecard41_result, glucose30_time, glucose30_result, glucose41_time, glucose41_result, glucose421_time, glucose421_result, lingeling_time, lingeling_result, maplechrono_time, maplechrono_result, maplecm_time, maplecm_result, maplesat_time, maplesat_result, mergesat30_time, mergesat30_result, minicard_time, minicard_result, minisat22_time, minisat22_result, minisat-gh_time, minisat-gh_result, filter_type, order_upper, accuracy, adder_count, wordlength, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound\n")

    results = []
    for i in range(1):
        print("running test: ", i)
        params = generate_random_filter_params()
        filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength, accuracy, upper_cutoff, lower_cutoff, passband_upperbound, passband_lowerbound, stopband_upperbound, stopband_lowerbound = params
        pysat = FIRFilterPysat(filter_type, order_upper, freqx_axis, freq_upper, freq_lower, ignore_lowerbound_lin, adder_count, wordlength)

        result_list = pysat.runsolver(timeout=timeout)  # Adjust the timeout as needed
        results.append((result_list, *params))
        with open("pysat_solver_bench.txt", "a") as file:
            file.write(f"{result_list}, {filter_type}, {order_upper}, {accuracy}, {adder_count}, {wordlength}, {upper_cutoff}, {lower_cutoff}, {passband_upperbound}, {passband_lowerbound}, {stopband_upperbound}, {stopband_lowerbound}\n")
        print("test ", i, " is completed")

    print("Benchmark completed and results saved to pysat_solver_bench.txt")
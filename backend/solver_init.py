from z3 import *
import numpy as np
import functools
from .solver_func import SolverFunc

class SolverInit():
    def __init__(self):
        self.freq_upper = np.array([])
        self.freq_lower = np.array([])
        self.freqx_axis = np.array([])

        self.filter_type = None
        self.order_upper = None
        self.order_lower = None
        self.order_current = None
        self.wordlength  = None
        self.h_int_res = []
        self.ignore_lowerbound=-30
        self.ignore_lowerbound_lin=10**(self.ignore_lowerbound / 20)

    def set_input_arg(self, input_list):
        # if not isinstance(input_list, (list, tuple)) or len(input_list) != 7:
        #     raise TypeError("input must be a list or tuple with 7 elements")
        self.filter_type = input_list[0]
        self.order_upper = input_list[1]
        self.order_lower = input_list[2]
        self.wordlength = input_list[3]
        self.freq_upper = input_list[4]
        self.freq_lower = input_list[5]
        self.freqx_axis = input_list[6]

    def sum_float_handler(self,lst):
        return functools.reduce(lambda a, b: a + b, lst, 0)

    
    def runsolver(self):
        self.order_current = int(self.order_upper)
        

        print ("solver called")
        #initiate solver_func
        sf=SolverFunc(self.filter_type, self.order_current)
    

        
        print("filter order: ", self.order_current)
        print("ignore lower then:", self.ignore_lowerbound_lin)
        #linearize the bounds
        self.freq_upper_lin=sf.db_to_linear(self.freq_upper)
        self.freq_lower_lin=sf.db_to_linear(self.freq_lower)

        

        #declaring variables
        h_int = [Int(f'h_int_{i}') for i in range(self.order_current)]

        # Create a Z3 solver instance
        solver = Solver()

        # Create the sum constraints
        for i in range(len(self.freqx_axis)):
            print("upper freq: ",self.freq_upper_lin[i])
            print("lower freq: ",self.freq_lower_lin[i])

            print("freq: ",self.freqx_axis[i])
            term_sum_exprs = 0
            half_order=self.order_current//2
            if np.isnan(self.freq_upper_lin[i]) or np.isnan(self.freq_lower_lin[i]):
                continue            
            
            for j in range((self.order_current//2)):
                cm_const=sf.cm_handler(j, self.freqx_axis[i])
                term_sum_exprs+=h_int[j] * cm_const
            solver.add(term_sum_exprs<=self.freq_upper_lin[i])
            
            if self.freq_lower_lin[i] < self.ignore_lowerbound_lin:
                continue
            solver.add(term_sum_exprs>=self.freq_lower_lin[i])



        for i in range(self.order_current // 2):
            mirror = (i + 1) * -1

            if self.filter_type == 0 or self.filter_type == 1:
                solver.add(h_int[i] == h_int[mirror])
                print(f"Added constraint: h_int[{i}] == h_int[{mirror}]")

            if self.filter_type == 2 or self.filter_type == 3:
                solver.add(h_int[i] == -h_int[mirror])
                print(f"Added constraint: h_int[{i}] == -h_int[{mirror}]")

            print(f"stype = {self.filter_type}, {i} is equal with {mirror}")

        print ("solver running")
       
        if solver.check() == sat:
            print ("solver sat")
            model = solver.model()
            for i in range(self.order_current):
                print(f'h_int_{i} = {model[h_int[i]]}')
                self.h_int_res.append(model[h_int[i]].as_long())
            
            print(self.h_int_res)

        else:
            print("Unsatisfiable")

        print ("solver stopped")

    def get_result(self):
        return self.h_int_res


        




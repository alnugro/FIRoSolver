from z3 import *
import numpy as np
import functools
from .solver_func import SolverFunc

class SolverInit():
    def __init__(self):
        

        #input data from user
        self.filter_type = None
        self.order_upper = None
        self.order_lower = None
        self.wordlength = None
        self.sampling_rate = None

        self.frequency_bounds_dict = {}


        #variable that will be used in solver_init
        self.order_current = None
        self.h_int_res = []
        self.ignore_lowerbound=-30
        self.ignore_lowerbound_lin=10**(self.ignore_lowerbound / 20)

    #set input data
    def update_plotter_data(self, data_dict):
        for data_key, data_value in data_dict.items():
            if hasattr(self, data_key):
                setattr(self, data_key, data_value)

    def set_input_arg(self, frequency_bounds_dict):
        self.frequency_bounds_dict = frequency_bounds_dict

    def sum_float_handler(self,lst):
        return functools.reduce(lambda a, b: a + b, lst, 0)

    def run_solver(self):
        for bound_order in self.frequency_bounds_dict:
            self.run_z3(bound_order, self.frequency_bounds_dict[bound_order])


    def run_z3(self, filter_order, freq_bounds):
        freqx_axis, Magnitude_upper_bound, Magnitude_lower_bound = freq_bounds
        print("solver ran")



        




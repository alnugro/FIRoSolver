import random
import time
import numpy as np
import copy
from pebble import ProcessPool
from concurrent.futures import TimeoutError  # Correct import for TimeoutError
import multiprocessing
import math
import sys
from filelock import FileLock
import os

try:
    from .solver_func import SolverFunc
    from .bound_error_handler import BoundErrorHandler
    from .solver_presolve import Presolver
    from .solver_error_predictor import ErrorPredictor
    from .solver_main_problem import MainProblem
except:

    from solver_func import SolverFunc
    from bound_error_handler import BoundErrorHandler
    from solver_presolve import Presolver
    from solver_error_predictor import ErrorPredictor
    from solver_main_problem import MainProblem




class SolverBackend():
    def __init__(self, input_data):
        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.ignore_lowerbound = None
        self.wordlength = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None
        self.worker = None

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.pysat_thread = None
        self.z3_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

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
        
        #declare data that is local to backend
        self.input_data = input_data  


        self.sf = SolverFunc(self.input_data)
        self.bound_too_small_flag = False

        #increase accuracy
        self.presolver_accuracy_multiplier = 1.5

        self.xdata, self.upperbound_lin, self.lowerbound_lin = self.sf.interpolate_bounds_to_order(self.order_upperbound)
        self.xdata_presolve, self.upperbound_lin_presolve, self.lowerbound_lin_presolve = self.sf.interpolate_bounds_to_order(int(self.order_upperbound*self.presolver_accuracy_multiplier))

        
        #update input data with interpolated data
        self.input_data.update({
            'xdata' : self.xdata,
            'upperbound_lin': self.upperbound_lin,
            'lowerbound_lin': self.lowerbound_lin,
            'xdata_presolve': self.xdata_presolve,
            'upperbound_lin_presolve': self.upperbound_lin_presolve,
            'lowerbound_lin_presolve': self.lowerbound_lin_presolve
        })

        self.end_result = None

        if self.filter_type == 1 or self.filter_type == 3:
            self.half_order = (self.order_upperbound + 1) // 2
        elif self.filter_type == 0:
            self.half_order = (self.order_upperbound // 2) + 1
        else:
            self.half_order = self.order_upperbound// 2

        self.input_data.update({
            'half_order': self.half_order
        })

        self.xdata_temp = None
        self.upperbound_lin_temp = None
        self.lowerbound_lin_temp = None


    def db_to_linear(self,value):
        linear_value = 10 ** (value / 20)
        return linear_value
    
    def error_prediction(self):
        err_handler = ErrorPredictor(self.input_data)
        upperbound_lin_from_err, lowerbound_lin_from_err, h_res, gain_res= err_handler.run_error_prediction()
        
        if err_handler.error_predictor_canceled:
            return
        
        self.upperbound_lin = upperbound_lin_from_err
        self.lowerbound_lin = lowerbound_lin_from_err
        
        #flag that some of the upper-and lowerbounds becomes equal, so a strict constraint
        if err_handler.bound_too_small_flag:
            self.bound_too_small_flag = True
        
        
        self.input_data.update({
        'upperbound_lin': self.upperbound_lin,
        'lowerbound_lin': self.lowerbound_lin
        })

        print(f"error_pred: {h_res}")

        #patch up the bounds
        bound_patcher = BoundErrorHandler(self.input_data)
        leaks,leaks_mag = bound_patcher.leak_validator(h_res, gain_res)

        h_res2 = []
        gain_res2 = 0
        if leaks:
            #patch the input and lower the bound further if found
            xdata_temp,upperbound_lin_temp,lowerbound_lin_temp = bound_patcher.patch_bound_error(leaks,leaks_mag)

            #flag that some of the upper-and lowerbounds becomes equal, so strict constraint
            if bound_patcher.bound_too_small_flag:
                self.bound_too_small_flag = True
        
            
            #assign test run
            self.input_data.update({
                'xdata':xdata_temp,
                'upperbound_lin': upperbound_lin_temp,
                'lowerbound_lin': lowerbound_lin_temp
                })

            err_handler2 = ErrorPredictor(self.input_data)
            upperbound_lin_from_err2, lowerbound_lin_from_err2, h_res2 ,gain_res2= err_handler2.run_error_prediction(only_check_sat=True)
            
            if err_handler2.error_predictor_canceled == True:
                print("value from patcher is reverted due to unsat")
                #revert input again if the problem is unsat
                self.input_data.update({
                'xdata':self.xdata,
                'upperbound_lin': self.upperbound_lin,
                'lowerbound_lin': self.lowerbound_lin
                })
            else:
                self.xdata = xdata_temp
                self.upperbound_lin = upperbound_lin_temp
                self.lowerbound_lin = lowerbound_lin_temp
        else: 
            print("no leaks found")

        #uncomment if you want to check the solution for the error prediction
        # bound_patcher2 = BoundErrorHandler(self.input_data)
        # leaks,leaks_mag = bound_patcher2.leak_validator(h_res2, gain_res2)

    def find_best_adder_s(self, presolve_result):
        main = MainProblem(self.input_data)
        
        if self.gurobi_thread > 0:
            target_result, best_adderm, adder_s_h_zero_best= main.find_best_adder_s(presolve_result)
            
        
        else:
            target_result, best_adderm, adder_s_h_zero_best= main.find_best_adder_s_z3_paysat(presolve_result)
        
        half_adder_s = self.half_order - adder_s_h_zero_best - 1

        if self.filter_type == 0:
            adder_s = 2 * half_adder_s
        else:
            adder_s =( 2 * half_adder_s) + 1

        total_adder = adder_s + best_adderm

        if target_result is None:
            return None, best_adderm ,total_adder,adder_s_h_zero_best
        
        print(f"self.half_order {self.half_order}")
        print(f"max_zero {presolve_result['max_zero']}")
        print(f"total_adder {total_adder}")
        print(f"h {target_result}")
        print(f"h {target_result['h_res']}")
        print(f"best_adderm {best_adderm}")

        target_result.update({
            'total_adder': int(total_adder),
            'adder_m': int(best_adderm),
            'adder_s': int(adder_s),
            'half_adder_s': int(half_adder_s),
            'half_order':self.half_order,
            'wordlength': self.wordlength,
            'adder_wordlength': self.wordlength+ self.adder_wordlength_ext,
            'adder_depth': self.adder_depth,
            'fracw':self.wordlength-self.intW
        })


        return target_result, best_adderm ,total_adder,adder_s_h_zero_best
    
    def deep_search_adder_total(self,presolve_result, input_data_dict):
        main = MainProblem(self.input_data)
        target_result, best_adderm, h_zero_best = main.deep_search(presolve_result ,input_data_dict)
        total_adder = None
        adder_s = None
        if h_zero_best:
            half_adder_s = self.half_order - h_zero_best - 1
            if self.filter_type == 0:
                adder_s = 2 * half_adder_s
            else:
                adder_s =( 2 * half_adder_s) + 1
            total_adder = adder_s + best_adderm

        # print(f"self.half_order {self.half_order}")
        # print(f"h_zero_best {h_zero_best}")
        # print(f"h {target_result['h']}")
        # print(f"h_res {target_result['h_res']}")

        # print(f"total_adder {total_adder}")
        # print(f"best_adderm {best_adderm}")
            target_result.update({
                'total_adder': int(total_adder),
                'adder_m': int(best_adderm),
                'adder_s': int(adder_s),
                'half_adder_s': int(half_adder_s),
                'wordlength': self.wordlength,
                'adder_wordlength': self.wordlength+ self.adder_wordlength_ext,
                'adder_depth': self.adder_depth,
                'fracw':self.wordlength-self.intW
            })

        return target_result, best_adderm, total_adder, h_zero_best


    def find_best_adder_m(self,presolve_result):
        main = MainProblem(self.input_data)
        target_result, best_adderm, adderm_h_zero_best = main.find_best_adder_m(presolve_result)

        half_adder_s = self.half_order - adderm_h_zero_best - 1

        if self.filter_type == 0:
            adder_s = 2 * half_adder_s
        else:
            adder_s =( 2 * half_adder_s) + 1

        total_adder = adder_s + best_adderm

        print(f"self.half_order {self.half_order}")
        print(f"h_zero_best {adderm_h_zero_best}")
        print(f"h {target_result['h']}")
        print(f"h_res {target_result['h_res']}")
        print(f"total_adder {total_adder}")
        print(f"best_adderm {best_adderm}")

        target_result.update({
            'total_adder': int(total_adder),
            'adder_m': int(best_adderm),
            'adder_s': int(adder_s),
            'half_adder_s': int(half_adder_s),
            'half_order':self.half_order,
            'wordlength': self.wordlength,
            'adder_wordlength': self.wordlength+ self.adder_wordlength_ext,
            'adder_depth': self.adder_depth,
            'fracw':self.wordlength-self.intW
        })
        return target_result, best_adderm, total_adder, adderm_h_zero_best
    

    def deep_search_adder_total_old(self,presolve_result, input_data_dict):
        main = MainProblem(self.input_data)
        target_result, best_adderm, h_zero_best = main.deep_search(presolve_result ,input_data_dict)
        total_adder = None
        adder_s = None
        if h_zero_best:
            half_adder_s = self.half_order - h_zero_best - 1
            if self.filter_type == 0:
                adder_s = 2 * half_adder_s
            else:
                adder_s =( 2 * half_adder_s) + 1
            total_adder = adder_s + best_adderm

        # print(f"self.half_order {self.half_order}")
        # print(f"h_zero_best {h_zero_best}")
        # print(f"h {target_result['h']}")
        # print(f"h_res {target_result['h_res']}")

        # print(f"total_adder {total_adder}")
        # print(f"best_adderm {best_adderm}")
            target_result.update({
                'total_adder': int(total_adder),
                'adder_m': int(best_adderm),
                'adder_s': int(adder_s),
                'half_adder_s': int(half_adder_s),
                'wordlength': self.wordlength,
                'adder_wordlength': self.wordlength+ self.adder_wordlength_ext,
                'adder_depth': self.adder_depth,
                'fracw':self.wordlength-self.intW
            })

        return target_result, best_adderm, total_adder, h_zero_best

    def solving_result_barebone(self,presolve_result,adderm,h_zero_count):
        main = MainProblem(self.input_data)
        if self.gurobi_thread > 0:
            target_result_best, satisfiability = main.try_asserted(presolve_result,adderm,h_zero_count)

        else:
            target_result_best, satisfiability = main.try_asserted_z3_pysat(adderm,h_zero_count)

        adder_s = self.half_order - h_zero_count - 1
        total_adder = (2*adder_s) - 1 + adderm if self.filter_type == 0 or self.filter_type == 2 else (2*adder_s) + adderm
        target_result_best.update({
            'total_adder': total_adder,
            'adder_m':adderm,
            'adder_s': adder_s,
            'half_order':self.half_order,
            'wordlength': self.wordlength,
            'adder_wordlength': self.wordlength+ self.adder_wordlength_ext,
            'adder_depth': self.adder_depth,
            'fracw':self.wordlength-self.intW
        })

            
        return target_result_best, satisfiability


    def solver_presolve(self):
        #interpolate original data first
        presolve_result = None

        #always run presolver first
        presolver = Presolver(self.input_data)

        if self.gurobi_thread > 0:
            #if gurobi is available then use gurobi, because it is way faster to find the minimum solver order and can be used to find minmax variables
            presolve_result = presolver.run_presolve_gurobi()
            min_adderm = presolver.min_adderm_finder(presolve_result['hmax'],presolve_result['hmin'])
            print(f"min_adderm {min_adderm}")
            presolve_result.update({
                'min_adderm' : min_adderm,
                'min_adderm_without_zero' : 0,
            })
        else:
            presolve_result = presolver.run_presolve_z3_pysat()
        
        if presolve_result['hmax'] != None:
            # min_adderm = presolver.min_adderm_finder(presolve_result['hmax'],presolve_result['hmin'], False)
            # min_adderm_without_zero = presolver.min_adderm_finder(presolve_result['hmax_without_zero'],presolve_result['hmin_without_zero'],False)
            presolve_result.update({
                'min_adderm' : 0,
                'min_adderm_without_zero' : 0,
            })
        else:
            # min_adderm_without_zero = presolver.min_adderm_finder(presolve_result['h_res'], None ,True)
            presolve_result.update({
                'min_adderm' : None,
                'min_adderm_without_zero' : 0,
            })
        
        
        
        return presolve_result
    
    def result_validator(self,h_res,gain_res):
        #patch up the bounds
        bound_patcher = BoundErrorHandler(self.input_data)
        leaks,leaks_mag = bound_patcher.leak_validator(h_res, gain_res)

        # print(f"this is xdata before {(self.xdata)}")
        # print(f"this is upperbound_lin {(self.upperbound_lin)}")
        # print(f"this is lowerbound_lin {(self.lowerbound_lin)}")
        # print(f"this is xdata before {len(self.xdata)}")

        h_res2 = []
        gain_res2 = 0
        leak_flag = None

        if leaks:
            leak_flag = True

        else: 
            print("no leaks found")
            leak_flag = False

        return leaks, leaks_mag
    
    def patch_leak(self, leaks,leaks_mag):
        self.xdata_temp = self.xdata
        self.upperbound_lin_temp = self.upperbound_lin
        self.lowerbound_lin_temp = self.lowerbound_lin

        bound_patcher = BoundErrorHandler(self.input_data)
        #patch the input and lower the bound further if found
        self.xdata,self.upperbound_lin,self.lowerbound_lin = bound_patcher.patch_bound_error(leaks,leaks_mag)

        #flag that some of the upper-and lowerbounds becomes equal, so strict constraint
        if bound_patcher.bound_too_small_flag:
            self.bound_too_small_flag = True
    
        # print(f"this is leaks_mag {leaks_mag}")
        # print(f"this is xdata after {(self.xdata)}")
        # print(f"this is upperbound_lin {self.upperbound_lin_temp}")
        # print(f"this is lowerbound_lin {self.lowerbound_lin_temp}")
        # print(f"this is xdata after {len(self.xdata)}")

         
        #update dict
        self.input_data.update({
            'xdata':self.xdata,
            'upperbound_lin': self.upperbound_lin,
            'lowerbound_lin': self.lowerbound_lin
            })

    def revert_patch(self):
        self.xdata = self.xdata_temp 
        self.upperbound_lin = self.upperbound_lin_temp
        self.lowerbound_lin = self.lowerbound_lin_temp

        #update dict
        self.input_data.update({
            'xdata':self.xdata,
            'upperbound_lin': self.upperbound_lin,
            'lowerbound_lin': self.lowerbound_lin
            })

        
    
    def gurobi_test(self):
        try:
            from gurobipy import Model, GRB
            
            # Create a new model
            model = Model("test_model")

            # Create variables
            x = model.addVar(name="x")
            y = model.addVar(name="y")

            # Set objective: maximize 3x + 4y
            model.setObjective(3 * x + 4 * y, GRB.MAXIMIZE)

            # Add constraint: x + 2y <= 14
            model.addConstr(x + 2 * y <= 14, "c1")

            # Optimize the model
            model.optimize()

            # Check if an optimal solution was found
            if model.status == GRB.OPTIMAL:
                print("\nGurobi Test completed.......\n")

        
            else: 
                raise ImportError("Gurobi is somehow broken, simple test should be sat: Check your installation and probably contact Gurobi")

            

        except Exception as e:
            raise ImportError(f"Gurobi encountered an error, Check your installation: {e}")



    

if __name__ == "__main__":
    # Test inputs
    filter_type = 0
    order_current = 16 #this is literally the order
    accuracy = 4
    wordlength = 9
    gain_upperbound = 2.7
    gain_lowerbound = 1
    coef_accuracy = 3
    intW = 1

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 2

    gain_wordlength = 13
    gain_intW = 4

    gurobi_thread = 10
    pysat_thread = 0
    z3_thread = 0

    timeout = 0


    passband_error = 0.1
    stopband_error = 0.1
    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.2 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0

    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point - 1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point - 1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point - 1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point - 1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point - 1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point - 1])

    # Beyond this bound, lowerbound will be ignored
    ignore_lowerbound = -60

    # Linearize the bounds
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
    cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)

    input_data = {
        'filter_type': filter_type,
        'order_upperbound': order_current,
        'original_xdata': freqx_axis,
        'original_upperbound_lin': upperbound_lin,
        'original_lowerbound_lin': lowerbound_lin,
        'ignore_lowerbound': ignore_lowerbound_lin,
        'cutoffs_x': cutoffs_x,
        'cutoffs_upper_ydata_lin': cutoffs_upper_ydata_lin,
        'cutoffs_lower_ydata_lin': cutoffs_lower_ydata_lin,
        'wordlength': wordlength,
        'adder_count': adder_count,
        'adder_depth': adder_depth,
        'avail_dsp': avail_dsp,
        'adder_wordlength_ext': adder_wordlength_ext,  # This is extension, not the adder wordlength
        'gain_wordlength': gain_wordlength,
        'gain_intW': gain_intW,
        'gain_upperbound': gain_upperbound,
        'gain_lowerbound': gain_lowerbound,
        'coef_accuracy': coef_accuracy,
        'intW': intW,
        'gurobi_thread': gurobi_thread,
        'pysat_thread': pysat_thread,
        'z3_thread': z3_thread,
        'timeout': 0,
        'start_with_error_prediction': False,
        'solver_accuracy_multiplier': accuracy,
        'deepsearch': True,
        'patch_multiplier': 1,
        'gurobi_auto_thread': False,
        'seed': 0,
        'worker': 2,
        'search_step': 4,
        'problem_id': 0,
        'am_start': 0,

    }

    # Create an instance of SolverBackend
    backend = SolverBackend(input_data)

    # # backend.solver_presolve()
    # # backend.error_prediction()
    # backend.gurobi_test()

    # start presolve
    presolve_result = backend.solver_presolve()
    print(presolve_result)
    target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)
    target_result2, best_adderm2, total_adder2, adderm_h_zero_best = backend.find_best_adder_m(presolve_result)
    
    while True:
        leaks, leaks_mag = backend.result_validator(target_result['h_res'],target_result['gain'])
        
        if leaks:
            print("leak_flag")

            target_result, satisfiability = backend.solving_result_barebone(presolve_result,best_adderm,adder_s_h_zero_best)
            if satisfiability == 'unsat':
                print("problem is unsat from asserting the leak to the problem")
                break
        else:
            break
    
    print(target_result)

        
    #test main problem
    presolve_result = backend.solver_presolve()
    target_result, best_adderm ,total_adder, adder_s_h_zero_best = backend.find_best_adder_s(presolve_result)

    backend.result_validator(target_result['h_res'],target_result['gain'])

    # Packing variables into a dictionary
    data_dict = {
    'best_adderm_from_s': int(best_adderm),
    'total_adder_s': int(total_adder),
    'adder_s_h_zero_best': int(adder_s_h_zero_best),
    }

    if presolve_result['max_zero'] == 0:
        print("Deep Search canceled, no search space for h_zero: ")
    else:
        target_result3, best_adderm3, total_adder3, h_zero_best3 = backend.deep_search_adder_total2(presolve_result, data_dict)
    
    print(f"best_adderm {best_adderm}")
    print(f"total_adder_global_minimum {best_adderm3}")
    
    print(f"total_adder_s {total_adder}")
    print(f"total_adder_global_minimum {total_adder3}")

    # # .........test interpolation data..........
    # interp_xdata = backend.xdata
    # interp_upper = backend.upperbound_lin
    # interp_lower = backend.lowerbound_lin

    # import matplotlib.pyplot as plt

    # plt.scatter(freqx_axis, upperbound_lin, color='blue', marker='o', s=100)
    # plt.scatter(interp_xdata, interp_upper, color='red', marker='x', s=100)

    # # Add a title and labels to the axes
    # plt.title('Simple Scatter Plot')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')

    # # Display the plot
    # plt.show()




    

from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError, CancelledError, wait, ALL_COMPLETED
import traceback
import time
import copy
import numpy as np
import random

try:
    from .formulation_pysat import FIRFilterPysat
    from .formulation_z3_pbsat import FIRFilterZ3
    from .formulation_gurobi import FIRFilterGurobi
    from .solver_func import SolverFunc
except:
    from formulation_pysat import FIRFilterPysat
    from formulation_z3_pbsat import FIRFilterZ3
    from formulation_gurobi import FIRFilterGurobi
    from solver_func import SolverFunc

class ErrorPredictor:
    def __init__(self, input_data
                 ):
        
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

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.pysat_thread = None
        self.z3_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.cutoffs_upper_ydata_lin = None
        self.cutoffs_lower_ydata_lin = None

        self.seed = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)
        
        self.upperbound_lin_calc = None
        self.lowerbound_lin_calc = None
        
        self.input_data = input_data

        self.bound_too_small_flag = False
        self.error_predictor_canceled = False

    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        'order_upperbound': self.order_upperbound,
        }

        return input_data_sf
    
    def run_error_prediction(self, only_check_sat = False):
        if only_check_sat == False:
            if self.gurobi_thread > 0:
                upperbound_lin_calc, lowerbound_lin_calc, h_res_calc, gain_res_calc = self.gurobi_error_prediction(False)
            elif self.z3_thread > 0:
                upperbound_lin_calc, lowerbound_lin_calc, h_res_calc, gain_res_calc = self.z3_error_prediction(False)
            elif self.pysat_thread > 0:
                upperbound_lin_calc, lowerbound_lin_calc, h_res_calc, gain_res_calc = self.pysat_error_prediction(False)


        #now check if the problem is still satisfiable
        if self.gurobi_thread > 0:
            upperbound_lin_sat, lowerbound_lin_sat, h_res_sat,gain_res_sat = self.gurobi_error_prediction(True)
        elif self.z3_thread > 0:
            upperbound_lin_sat, lowerbound_lin_sat, h_res_sat,gain_res_sat = self.z3_error_prediction(True)
        elif self.pysat_thread > 0:
            upperbound_lin_sat, lowerbound_lin_sat, h_res_sat,gain_res_sat = self.pysat_error_prediction(True)
        
        # print(f"before {h_res_calc}")
        # print(f"after {h_res_sat}")
        if only_check_sat:
            if h_res_sat == None:
                print("Patcher Prediction cancelled, due to problem become unsat")
                self.error_predictor_canceled == True
            return upperbound_lin_sat, lowerbound_lin_sat, h_res_sat, gain_res_sat
           
        else:
            if h_res_sat == None:
                self.error_predictor_canceled == True
                print("Error Prediction cancelled, due to problem become unsat")

            return upperbound_lin_calc, lowerbound_lin_calc, h_res_sat, gain_res_sat
    
    
    def gurobi_error_prediction(self, sat_check):
        h_res = []
        freq_upper_lin = None
        freq_lower_lin = None
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            0, 
            self.wordlength,
            0,
            0,
            0,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW
        )

        target_result = gurobi_instance.run_barebone(self.gurobi_thread,None)
        
        satisfiability = target_result['satisfiability']
        h_res_loc = target_result['h_res']
        gain_res_loc = target_result['gain_res']
        
        if sat_check == False:
            print("\n\n i ran here.............................\n")
            if satisfiability == "unsat":
                raise ValueError("problem is unsat")
            freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res_loc,self.upperbound_lin, self.lowerbound_lin, 'gurobi', None)
        else:
            if satisfiability == "unsat":
                h_res_loc = None
            freq_upper_lin = self.upperbound_lin
            freq_lower_lin = self.lowerbound_lin

        return freq_upper_lin, freq_lower_lin, h_res_loc, gain_res_loc

    def z3_error_prediction(self,sat_check):
        h_res = []
        freq_upper_lin = None
        freq_lower_lin = None


        satisfiability, h_res_loc ,gain_res_loc= self.z3_instance_creator().run_barebone(self.seed)
        
        if sat_check == False:
            if satisfiability == "unsat":
                raise ValueError("problem is unsat")
            freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res_loc,self.upperbound_lin, self.lowerbound_lin, 'z3',gain_res_loc)
        else:
            if satisfiability == "unsat":
                h_res_loc = None
            freq_upper_lin = self.upperbound_lin
            freq_lower_lin = self.lowerbound_lin
        
        return freq_upper_lin, freq_lower_lin,h_res_loc, gain_res_loc
        
    def pysat_error_prediction(self,sat_check):
        h_res = []

        freq_upper_lin = None
        freq_lower_lin = None


        satisfiability, h_res_loc ,gain_res_loc= self.pysat_instance_creator().run_barebone(self.seed)
        
        if sat_check == False:
            if satisfiability == "unsat":
                raise ValueError("problem is unsat")
            freq_upper_lin, freq_lower_lin  = self.calculate_error(h_res_loc,self.upperbound_lin, self.lowerbound_lin, 'pysat',gain_res_loc)
        else:
            if satisfiability == "unsat":
                h_res_loc = None
            freq_upper_lin = self.upperbound_lin
            freq_lower_lin = self.lowerbound_lin

        return freq_upper_lin, freq_lower_lin,h_res_loc,gain_res_loc
        
    
    def calculate_error(self, h_res, freq_upper, freq_lower, solver ,gain = None):
        if solver == 'pysat':
            delta_coef = 2**-(self.wordlength-self.intW)
            delta_gain = 2**-(self.wordlength-self.intW)
        else:
            delta_coef = 10 ** - self.coef_accuracy
            delta_gain = 2**-(self.gain_wordlength-self.gain_intW)

        # print(f"len h_res{len(h_res)}")

        delta_h_res = 2**-(self.wordlength-self.intW)

        #init var
        freq_upper_with_error_pred = np.copy(freq_upper)
        freq_lower_with_error_pred = np.copy(freq_lower)
        sf = SolverFunc(self.get_solver_func_dict())

        half_order = (self.order_upperbound // 2) +1 if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2)

        for omega in range(len(self.xdata)):
            delta_omega = []
            omega_result = 0
            if np.isnan(freq_upper[omega]) or np.isnan(freq_lower[omega]):
                continue

            for m in range(half_order):
                #calculate const
                cm = sf.cm_handler(m, self.xdata[omega])
                z_result_temp = h_res[m] * cm
                
                #calculate error
                h_res_error = (delta_h_res/h_res[m])**2 if h_res[m] != 0 else 0
                cm_error = (delta_coef/cm)**2 if cm != 0 else 0
                z_error_temp = np.sqrt(h_res_error + cm_error)

                delta_omega.append(z_result_temp*z_error_temp)
                omega_result += z_result_temp
            delta_omega = np.array(delta_omega)
            delta_omega = np.square(delta_omega)
            delta_omega_result = np.sqrt(np.sum(delta_omega))

            
            if gain != None:
                omega_error = (delta_omega_result/omega_result)**2 if omega_result != 0 else 0
                gain_error = (delta_gain/gain)**2 if gain != 0 else 0
                delta_error_result = np.sqrt(omega_error+gain_error) 
            else:
                delta_error_result = delta_omega_result

            # print(f"\nError result {delta_error_result}")
            # print(f"Omega Error result {delta_omega_result}")
            # print(f"freq before {freq_upper[omega]}")

            


            freq_upper_with_error_pred[omega] = freq_upper[omega]-delta_error_result
            freq_lower_with_error_pred[omega] = freq_lower[omega]+delta_error_result

            if freq_upper_with_error_pred[omega] < freq_lower_with_error_pred[omega]:
                freq_upper_with_error_pred[omega] = (freq_upper_with_error_pred[omega] + freq_lower_with_error_pred[omega])/2
                freq_lower_with_error_pred[omega] = (freq_upper_with_error_pred[omega] + freq_lower_with_error_pred[omega])/2
                self.bound_too_small_flag = True
            # print(f"freq {freq_upper[omega]}")


        return freq_upper_with_error_pred,freq_lower_with_error_pred




    def gurobi_instance_creator(self):
        gurobi_instance = FIRFilterGurobi(
             self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            0, 
            self.wordlength,
            0,
            0,
            0,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW
        )
    
        return gurobi_instance

    def z3_instance_creator(self):
        z3_instance = FIRFilterZ3(
                    self.filter_type, 
                    self.order_upperbound, 
                    self.xdata, 
                    self.upperbound_lin, 
                    self.lowerbound_lin, 
                    self.ignore_lowerbound, 
                    0, 
                    self.wordlength, 
                    0,
                    0,
                    0,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.coef_accuracy,
                    self.intW,
                    self.gain_wordlength,
                    self.gain_intW
                    )
        
        return z3_instance

    def pysat_instance_creator(self):
        pysat_instance = FIRFilterPysat(
                    self.filter_type, 
                    self.order_upperbound, 
                    self.xdata, 
                    self.upperbound_lin,
                    self.lowerbound_lin,
                    self.ignore_lowerbound, 
                    0, 
                    self.wordlength, 
                    0,
                    0,
                    0,
                    self.gain_upperbound,
                    self.gain_lowerbound,
                    self.intW
                    )
        
        return pysat_instance



if __name__ == "__main__":
   # Test inputs
    filter_type = 0
    order_current = 18
    accuracy = 4
    wordlength = 11
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 5
    intW = 1

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 4

    gain_wordlength = 13
    gain_intW = 4

    gurobi_thread = 10
    pysat_thread = 0
    z3_thread = 0

    timeout = 0


    passband_error = 0.09492187500000002
    stopband_error = 0.09492187500000002
    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
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
        'seed': 0
    }

    h_res = [192,128,0,0,0]

    # Create an instance of SolverBackend
    err_handler = ErrorPredictor(input_data)
    # print(f"before: {err_handler.lowerbound_lin}")
    # print(f"before: {err_handler.lowerbound_lin}")
    # print(f"before: {err_handler.lowerbound_lin}")

    err_handler.run_error_prediction()
    
    # err_handler.gurobi_error_prediction(1)
    # err_handler.z3_error_prediction(1)
    # err_handler.pysat_error_prediction(0)



    # err_handler.execute_parallel_error_prediction()
    # print(f"After: {err_handler.lowerbound_lin}")
    # print(f"After: {err_handler.lowerbound_lin}")
    # print(f"After: {err_handler.lowerbound_lin}")
   

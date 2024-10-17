
import numpy as np

from backend.solver_func import SolverFunc
from backend.backend_main import SolverBackend
from backend.formulation_gurobi import FIRFilterGurobi

# try:
#     from .formulation_pysat import FIRFilterPysat
#     from .formulation_z3_pbsat import FIRFilterZ3
#     from .formulation_gurobi import FIRFilterGurobi
#     from .solver_func import SolverFunc

# except ImportError:
#     from formulation_pysat import FIRFilterPysat
#     from formulation_z3_pbsat import FIRFilterZ3
#     from formulation_gurobi import FIRFilterGurobi
#     from solver_func import SolverFunc
#     from backend_main import SolverBackend


class QuickCheck:
    def __init__(self, input_data):
        
        # Explicit declaration of instance variables with default values (if applicable)
        self.filter_type = None
        self.order_upperbound = None

        self.wordlength = None
        
        self.ignore_lowerbound = None
        self.gain_upperbound = None
        self.gain_lowerbound = None
        self.coef_accuracy = None
        self.intW = None

        self.gain_wordlength = None
        self.gain_intW = None

        self.gurobi_thread = None
        self.z3_thread = None
        self.pysat_thread = None

        self.timeout = None
        self.start_with_error_prediction = None

        self.xdata = None
        self.upperbound_lin = None
        self.lowerbound_lin = None

        self.adder_count = None
        self.adder_depth = None
        self.avail_dsp = None
        self.adder_wordlength_ext = None

        # Dynamically assign values from input_data, skipping any keys that don't have matching attributes
        for key, value in input_data.items():
            if hasattr(self, key):  # Only set attributes that exist in the class
                setattr(self, key, value)

        self.sf = SolverFunc(input_data)
        self.bound_too_small_flag = False

        self.xdata, self.upperbound_lin, self.lowerbound_lin = self.sf.interpolate_bounds_to_order(self.order_upperbound)



        self.order_current = None

        self.min_order = None
        self.min_gain = None
        self.h_res = None
        self.gain_res = None

        self.gain_res = None
        self.h_res = None
        self.satisfiability = 'unsat'

        self.half_order = (self.order_upperbound // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2) - 1
        self.max_zero_reduced = 0

        self.result_model = {}

    def check_sat(self, option = None ,h_zero = 0):
        gurobi_instance = FIRFilterGurobi(
            self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            self.adder_count, 
            self.wordlength, 
            self.adder_depth,
            self.avail_dsp,
            self.adder_wordlength_ext,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW)
        

        if h_zero > 0 or option != None:
            target_result = gurobi_instance.run_barebone(self.gurobi_thread, option , h_zero)

        else:
            target_result = gurobi_instance.run_barebone(self.gurobi_thread, None , None)
        satisfiability = target_result['satisfiability']

        return target_result,satisfiability
    

    def check_sat_real(self, option = None ,h_zero = 0):
        gurobi_instance = FIRFilterGurobi(
           self.filter_type, 
            self.order_upperbound, #you pass upperbound directly to gurobi
            self.xdata, 
            self.upperbound_lin, 
            self.lowerbound_lin, 
            self.ignore_lowerbound, 
            self.adder_count, 
            self.wordlength, 
            self.adder_depth,
            self.avail_dsp,
            self.adder_wordlength_ext,
            self.gain_upperbound,
            self.gain_lowerbound,
            self.coef_accuracy,
            self.intW)

        if h_zero > 0 or option != None:
            target_result = gurobi_instance.run_barebone_real(self.gurobi_thread, option , h_zero)
        else:
            target_result = gurobi_instance.run_barebone_real(self.gurobi_thread, None , None)

        satisfiability = target_result['satisfiability']

        return  target_result,satisfiability
    

if __name__ == '__main__':
    # Unpack the dictionary to corresponding variables
    test_run = {
        1: {#S1
            'filter_type': 1,
            'order_current': 29,
            'accuracy': 3,
            'wordlength': 11,
            'gain_upperbound': 2.5,
            'gain_lowerbound': 1,
            'coef_accuracy': 3,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 6,
            'gain_intW': 2,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':  0.00636,
            'stopband_error':  0.00636,
            'lower_cutoff': 0.3,
            'upper_cutoff': 0.5,
        },
        2: {#S2
            'filter_type': 1,
            'order_current': 55,
            'accuracy': 3,
            'wordlength': 12,
            'gain_upperbound': 10.5,
            'gain_lowerbound': 1,
            'coef_accuracy': 5,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 9,
            'gain_intW': 5,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':  0.026,
            'stopband_error':   0.001,
            'lower_cutoff': 0.042,
            'upper_cutoff': 0.14,
        },
        3: {#L2
            'filter_type': 0,
            'order_current': 64,
            'accuracy': 3,
            'wordlength': 14,
            'gain_upperbound': 4.2,
            'gain_lowerbound': 1,
            'coef_accuracy': 6,
            'intW': 4,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 2,
            'gain_intW': 4,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':   0.02800,
            'stopband_error':   0.001,
            'lower_cutoff': 0.2,
            'upper_cutoff': 0.28,
        },
        4: {#X1
            'filter_type': 0,
            'order_current': 16,
            'accuracy': 4,
            'wordlength': 13,
            'gain_upperbound': 1.7,
            'gain_lowerbound': 1,
            'coef_accuracy': 6,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 6,
            'gain_intW': 2,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':  0.0001,
            'stopband_error':  0.0001,
            'lower_cutoff': 0.2,
            'upper_cutoff': 0.8,
        },
        5: {#G1
            'filter_type': 0,
            'order_current': 18,
            'accuracy': 6,
            'wordlength': 9,
            'gain_upperbound': 2.65,
            'gain_lowerbound': 1,
            'coef_accuracy': 6,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 5,
            'gain_wordlength': 6,
            'gain_intW': 3,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':  0.01,
            'stopband_error':  0.01,
            'lower_cutoff': 0.2,
            'upper_cutoff': 0.5,
        },
        6: {#Y1
            'filter_type': 1,
            'order_current': 31,
            'accuracy': 4,
            'wordlength': 13,
            'gain_upperbound': 2.6,
            'gain_lowerbound': 1,
            'coef_accuracy': 6,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 6,
            'gain_intW': 2,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':  0.00316,
            'stopband_error':  0.00316,
            'lower_cutoff': 0.3,
            'upper_cutoff': 0.5,
        },
        7: {#Y2
            'filter_type': 1,
            'order_current': 41,
            'accuracy': 3,
            'wordlength': 14,
            'gain_upperbound': 2.65,
            'gain_lowerbound': 1,
            'coef_accuracy': 6,
            'intW': 2,
            'adder_count': None,
            'adder_depth': 0,
            'avail_dsp': 0,
            'adder_wordlength_ext': 4,
            'gain_wordlength': 6,
            'gain_intW': 2,
            'gurobi_thread': 16,
            'pysat_thread': 0,
            'z3_thread': 0,
            'timeout': 0,
            'passband_error':   0.00115,
            'stopband_error':   0.00115,
            'lower_cutoff': 0.3,
            'upper_cutoff': 0.5,
        }
    }
    test_key = 3
    print(test_run[test_key])
    # Accessing the dictionary for the test_key 1 and assigning variables
    filter_type = test_run[test_key]['filter_type']
    order_current = test_run[test_key]['order_current']
    accuracy = test_run[test_key]['accuracy']
    wordlength = test_run[test_key]['wordlength']
    gain_upperbound = test_run[test_key]['gain_upperbound']
    gain_lowerbound = test_run[test_key]['gain_lowerbound']
    coef_accuracy = test_run[test_key]['coef_accuracy']
    intW = test_run[test_key]['intW']
    adder_count = test_run[test_key]['adder_count']
    adder_depth = test_run[test_key]['adder_depth']
    avail_dsp = test_run[test_key]['avail_dsp']
    adder_wordlength_ext = test_run[test_key]['adder_wordlength_ext']
    gain_wordlength = test_run[test_key]['gain_wordlength']
    gain_intW = test_run[test_key]['gain_intW']
    gurobi_thread = test_run[test_key]['gurobi_thread']
    pysat_thread = test_run[test_key]['pysat_thread']
    z3_thread = test_run[test_key]['z3_thread']
    timeout = test_run[test_key]['timeout']
    
    passband_error = test_run[test_key]['passband_error']
    stopband_error = test_run[test_key]['stopband_error']
    
    lower_cutoff = test_run[test_key]['lower_cutoff']
    upper_cutoff = test_run[test_key]['upper_cutoff']



    space = order_current * accuracy * 20 #original accuracy
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)


    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(lower_cutoff*(space))
    upper_half_point = int(upper_cutoff*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0 #will be ignored


    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(0)
    cutoffs_x.append(lower_cutoff)
    cutoffs_x.append(upper_cutoff)
    cutoffs_x.append(1)

    cutoffs_upper_ydata.append(1 + passband_error)
    cutoffs_upper_ydata.append(1 + passband_error)
    cutoffs_upper_ydata.append(0 + stopband_error)
    cutoffs_upper_ydata.append(0 + stopband_error)

    cutoffs_lower_ydata.append(1 - passband_error)
    cutoffs_lower_ydata.append(1 - passband_error)
    cutoffs_lower_ydata.append(0)
    cutoffs_lower_ydata.append(0)


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -100

    #linearize the bound
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = np.copy(cutoffs_upper_ydata)
    cutoffs_lower_ydata_lin = np.copy(cutoffs_lower_ydata)

    # print(np.array(upperbound_lin).tolist())
    # print(np.array(lowerbound_lin).tolist())
    # print(ignore_lowerbound)
    
    


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
        'adder_wordlength_ext': adder_wordlength_ext, #this is extension not the adder wordlength
        'gain_wordlength' : gain_wordlength,
        'gain_intW' : gain_intW,
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
        'deepsearch': False,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
    }

    quickie = QuickCheck(input_data)
    # target_result,_= quickie.check_sat_real('find_max_zero', 0)
    
    # while True:
    #     target_result2,_ = quickie.check_sat('try_h_zero_count',target_result['max_h_zero'])
    #     if target_result2['satisfiability'] == 'unsat':
    #         print(quickie.wordlength)
    #         break
    #     else:
    #         quickie.wordlength -= 1


    target_result2,_ = quickie.check_sat('findax_zero',0)
    adder_s = (quickie.half_order - target_result2['max_h_zero'] -1)*2
    print(adder_s)
    print(target_result2['satisfiability'])
    backend = SolverBackend(input_data)
    backend.result_validator(target_result2['h_res'],target_result2['gain_res'])
import numpy as np
import json
import os
import sys

try:
    
    from pydsp.circuit import Circuit
    from pydsp.modules import *
    from pydsp.io_utility import *
    from pydsp.file_writers import VHDLWriter
    from .live_logger import LiveLogger


except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pydsp.circuit import Circuit
    from pydsp.modules import *
    from pydsp.io_utility import *
    from pydsp.file_writers import VHDLWriter
    from live_logger import LiveLogger



class ResultHandler():
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

    def logger(self, logger_instance):
        return logger_instance

    def create_pydsp_circuit(self, result_model, validate_flag = False):
        # Unpack the dictionary values into the respective variables
        h_res = result_model['h_res']
        gain_res = result_model['gain']
        h = result_model['h']
        alpha = result_model['alpha']
        beta = result_model['beta']
        gamma = result_model['gamma']
        delta = result_model['delta']
        epsilon = result_model['epsilon']
        zeta = result_model['zeta']
        phi = result_model['phi']
        theta = result_model['theta']
        iota = result_model['iota']
        
        # Ensure that the lengths are the same
        adder_count = [len(alpha), len(beta), len(gamma), len(delta), len(epsilon), len(zeta), len(theta)-2]
        # Check if all lengths are the same
        if len(set(adder_count)) != 1:
            raise ValueError(f"Adder_counts inconsistent detected lengths: {adder_count}")
            return
        else:
            print(f"Adder_counts have consistent length: {adder_count[0]}")
            adder_multiplier_length = adder_count[0]

        half_order_count = [len(phi), len(h_res),len(iota)]

        # Check if all lengths are the same in the second group
        if len(set(half_order_count)) != 1:
            raise ValueError(f"Inconsistent lengths detected in half_order_count: {half_order_count}")
            return
        else:
            print(f"half_order_count have consistent length: {half_order_count[0]}")
            half_order_length= half_order_count[0]


        gamma_shift_value = []
        for i, gamma_index in enumerate(gamma):
            for j, gamma_val in enumerate(gamma[i]):
                if gamma_val == 1:
                    gamma_shift_value.append(j)
        
        zeta_shift_value = []
        for i, zeta_index in enumerate(zeta):
            for j, zeta_val in enumerate(zeta[i]):
                if zeta_val == 1:
                    zeta_shift_value.append(j)

        phi_shift_value = []
        for i, phi_index in enumerate(phi):
            for j, phi_val in enumerate(phi[i]):
                if phi_val == 1:
                    phi_shift_value.append(j)
        print(phi_shift_value)

        theta_connection = { m: None for m in range(half_order_length)}
        for i, theta_index in enumerate(theta):
            for j, theta_val in enumerate(theta[i]):
                if theta_val == 1:
                    theta_connection.update({
                        j : i 
                    })


        print(f"gamma_shift_value{gamma_shift_value}")
        print(f"zeta_shift_value{zeta_shift_value}")
        print(f"theta_connection{theta_connection}")


        # Additional variables
        total_adder = result_model['total_adder']
        best_adderm = result_model['adder_m']
        adder_s = result_model['adder_s']
        half_order = result_model['half_order']
        wordlength = result_model['wordlength']
        adder_wordlength = result_model['adder_wordlength']
        adder_depth = result_model['adder_depth']
        fracw = result_model['fracw']
        wordlength = result_model['wordlength']

        wordlength_ext = adder_wordlength - wordlength

        print(f"adder_wordlength{adder_wordlength}")



        c = [None for i in range(adder_multiplier_length+2)]
        c_out = [None for i in range(adder_multiplier_length+2)]
        l = [None for i in range(1, adder_multiplier_length+1)]
        r = [None for i in range(1, adder_multiplier_length+1)]
        s = [None for i in range(1, adder_multiplier_length+1)]
        w = [None for i in range(1, adder_multiplier_length+1)]
        x = [None for i in range(1, adder_multiplier_length+1)]
        y = [None for i in range(1, adder_multiplier_length+1)]
        z = [None for i in range(1, adder_multiplier_length+1)]
        
        a = [None for i in range(half_order)]
        b = [None for i in range(half_order)]
        o = [None for i in range(half_order)]
        h_ext = [None for i in range(half_order)]
        h_const = [None for i in range(half_order)]
        h_out = [None for i in range(half_order)]
        h_res_output = [None for i in range(half_order)]

        

        # create a circuit for a simple constant multiplication
        circuit = Circuit()
        # define data type used within the circuit => 8 bit signed numbers
        dat = DataType.SIGNED
      
        i_min_wordlength = -(2**(wordlength-1))
        i_max_wordlength = -i_min_wordlength-1

        i_min_adder_wordlength = -(2**(adder_wordlength-1))
        i_max_adder_wordlength = -i_min_adder_wordlength-1

        # create the input for our circuit
        c[0] = circuit.add(Input(input_ranges=[[i_min_adder_wordlength, i_max_adder_wordlength]], data_type=dat, name="c0"))
        c[-1] = circuit.add(Constant(value = 0, data_type=dat, name="c_last_zero"))

        for i in range(1, adder_multiplier_length+1):
            # c[i] = circuit.add(Input(input_ranges=[[i_min_adder_wordlength, i_max_adder_wordlength]], data_type=dat, name=f"c{i}"))
            for j, alpha_val in enumerate(alpha[i-1]):
                if alpha_val == 1:
                    s[i-1] = circuit.add(ShiftLeft(input_ranges=[c[j].output_range()], shift_length=gamma_shift_value[i-1], data_type=dat, name=f"alpha_shift{i}"))
                    circuit.connect(src_module=c[j], dst_module=s[i-1], dst_port=0)

            if epsilon[i-1] == 1:
                #do a substraction
                if delta[i-1] == 1:
                    for j, beta_val in enumerate(beta[i-1]):
                        if beta_val == 1:
                            z[i-1] = circuit.add(Sub(input_ranges=[s[i-1].output_range(), c[j].output_range()], data_type=dat, name=f"Sub{i}"))
                            circuit.connect(src_module=s[i-1], dst_module=z[i-1], dst_port=0)
                            circuit.connect(src_module=c[j], dst_module=z[i-1], dst_port=1)
                else:
                    for j, beta_val in enumerate(beta[i-1]):
                        if beta_val == 1:
                            z[i-1] = circuit.add(Sub(input_ranges=[c[j].output_range(), s[i-1].output_range()], data_type=dat, name=f"Sub{i}"))
                            circuit.connect(src_module=c[j], dst_module=z[i-1], dst_port=0)
                            circuit.connect(src_module=s[i-1], dst_module=z[i-1], dst_port=1)
            else:
                #do a addition
                for j, beta_val in enumerate(beta[i-1]):
                    if beta_val == 1:
                        z[i-1] = circuit.add(Add(input_ranges=[s[i-1].output_range(), c[j].output_range()], data_type=dat, name=f"Add{i}"))
                        circuit.connect(src_module=s[i-1], dst_module=z[i-1], dst_port=0)
                        circuit.connect(src_module=c[j], dst_module=z[i-1], dst_port=1)

            
            c[i] = circuit.add(ShiftRight(input_ranges=[z[i-1].output_range()], shift_length=zeta_shift_value[i-1], data_type=dat, name=f"zeta_shift{i}"))
            circuit.connect(src_module=z[i-1], dst_module=c[i], dst_port=0)
                

        for m in range(half_order):
            if iota[i-1] == 1:
                if theta_connection[m] == len(c)-1:
                    #if h_res zero skip everything, keep h_ext None at that particular index
                    continue
                #use addergraph
                h_ext[m] = circuit.add(ShiftLeft(input_ranges=[c[theta_connection[m]].output_range()], data_type=dat, name=f"phi_Shift{m}", shift_length=phi_shift_value[m]))
                circuit.connect(src_module=c[theta_connection[m]], dst_module=h_ext[m], dst_port=0)

            else:
                #use dsp
                h_res_val = int(h_res[m]*fracw)
                h_const[m] = circuit.add(Constant(value = h_res_val, data_type=dat, name=f"h_const{m}"))

                o[m] = circuit.add(Mult(input_ranges=[c[0].output_range(), h_const[m].output_range()], data_type=dat, name=f"Mult{m}"))
                circuit.connect(src_module=c[0], dst_module=o[m], dst_port=0)
                circuit.connect(src_module=h_const[m], dst_module=o[m], dst_port=0)

                
                h_ext[m] = circuit.add(ShiftLeft(input_ranges=[o[m].output_range()], data_type=dat, name=f"phi_Shift{m}", shift_length=phi_shift_value[m]))
                circuit.connect(src_module=o[m], dst_module=h_ext[m], dst_port=0)

            if validate_flag:
                h_res_output[m] = circuit.add(Output(input_ranges=[h_ext[m].output_range()], data_type=dat, name=f"Y{m}"))
                circuit.connect(src_module=h_ext[m], dst_module=h_res_output[m], dst_port=0)

        # circuit.print_info()
        if validate_flag:
            circuit.validate()
            c[0].define_input_data([1*2**fracw])
            for time_step in range(1):
                for m in range(half_order):
                    raw_h_val = h_res_output[m].get_output(time_step) if h_res_output[m] != None else 0
                    h_val = to_fixed_point(raw_h_val, fracw)
                    c0_val = c[0].get_output(time_step)
                    if h_val == h_res[m]:
                        print(f"h_res is validated => {h_res[m]} == {h_val}, input = {c0_val}")
                    else:
                        print(f"h_res is wrong => {h_res[m]} != {h_val}, input = {c0_val}")
                        return False
            return True
        


        h_final = [h_val for h_val in h_ext if h_val is not None]
        reg = [None for i in range(len(h_final)-1)]
        radd = [None for i in range(len(h_final)-1)]
        for i in range(len(reg)):
            reg[i] = circuit.add(Register(input_ranges=[h_final[i].output_range()], data_type=dat, name=f"Reg{i}"))
            circuit.connect(src_module=h_final[i], dst_module=reg[i], dst_port=0)

            radd[i] = circuit.add(Add(input_ranges=[reg[i].output_range(), h_final[i+1].output_range()], data_type=dat, name=f"Radd{i}"))
            circuit.connect(src_module=reg[i], dst_module=radd[i], dst_port=0)
            circuit.connect(src_module=h_final[i+1], dst_module=radd[i], dst_port=1)

        filter_out = circuit.add(Output(input_ranges=[radd[-1].output_range()], data_type=dat, name=f"filter_out"))
        circuit.connect(src_module=radd[-1], dst_module=filter_out, dst_port=0)

        circuit.validate()


        circuit.print_info()

        VHDLWriter.write("test.vhdl", circuit)


    def flip_filter_array(self, arr):
        arr_np = np.array(arr)
        if len(arr_np) % 2 == 0:
            # Even length: reverse the array excluding the first element (so no repeat of 0) and append the original array
            result = np.concatenate((arr_np[::-1], arr_np[1:]))
        else:
            # Odd length: reverse the array and append the original array
            result = np.concatenate((arr_np[::-1], arr_np))
        return result
   

    def load_with_lock(self, file_lock ,valid_flag = True):
        if valid_flag:
            filename='result_valid.json'
        else:
            filename='result_leak.json'

        with file_lock:  # Acquire lock before file access
            with open(filename, 'r') as json_file:
                loaded_data = json.load(json_file)

            # print(loaded_data)
            # Unpacking key-value pairs in a loop
            # for key, value in loaded_data.items():
            #     print(f"Key: {key}, Value: {value}")
        
        return loaded_data

if __name__ == '__main__':
    from backend_mediator import BackendMediator

      # Test inputs
    filter_type = 0
    order_current = 20
    accuracy = 3
    wordlength = 14
    gain_upperbound = 2
    gain_lowerbound = 1
    coef_accuracy = 10
    intW = 4

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 2
    intW = 4

    gain_wordlength = 6
    gain_intW = 2

    gurobi_thread = 10
    pysat_thread = 0
    z3_thread = 0

    timeout = 0

    space = order_current * accuracy * 50
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space) #according to Mr. Kumms paper
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3*(space))
    upper_half_point = int(0.6*(space))
    end_point = space

    freq_upper[0:lower_half_point] = 5
    freq_lower[0:lower_half_point] = 0

    freq_upper[upper_half_point:end_point] = -1
    freq_lower[upper_half_point:end_point] = -1000


    cutoffs_x = []
    cutoffs_upper_ydata = []
    cutoffs_lower_ydata = []

    cutoffs_x.append(freqx_axis[0])
    cutoffs_x.append(freqx_axis[lower_half_point-1])
    cutoffs_x.append(freqx_axis[upper_half_point])
    cutoffs_x.append(freqx_axis[end_point-1])

    cutoffs_upper_ydata.append(freq_upper[0])
    cutoffs_upper_ydata.append(freq_upper[lower_half_point-1])
    cutoffs_upper_ydata.append(freq_upper[upper_half_point])
    cutoffs_upper_ydata.append(freq_upper[end_point-1])

    cutoffs_lower_ydata.append(freq_lower[0])
    cutoffs_lower_ydata.append(freq_lower[lower_half_point-1])
    cutoffs_lower_ydata.append(freq_lower[upper_half_point])
    cutoffs_lower_ydata.append(freq_lower[end_point-1])


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -40

    #linearize the bound
    upperbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_upper]
    lowerbound_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in freq_lower]
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)

    cutoffs_upper_ydata_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_upper_ydata]
    cutoffs_lower_ydata_lin = [10 ** (f / 20) if not np.isnan(f) else np.nan for f in cutoffs_lower_ydata]


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
        'deepsearch': True,
        'patch_multiplier' : 1,
        'gurobi_auto_thread': False
    }
    


    result = ResultHandler(input_data)
    backend = BackendMediator(input_data)



    loaded_data = result.load_with_lock(backend.file_lock, False)

    for key_subitem, value_subitem in loaded_data['0'].items():
            print(f"Key:{key_subitem}, Value: {value_subitem}")

    result.create_pydsp_circuit(loaded_data['0'], False)

    # for key, value in loaded_data.items():
    #     for key_subitem, value_subitem in loaded_data[key].items():
    #         print(f"Key:{key}, Value: {value_subitem}")

    
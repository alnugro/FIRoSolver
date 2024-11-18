import numpy as np
import json
import os
import sys
from filelock import FileLock

import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTableWidget, 
    QTableWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer

try:
    
    from pydsp_local.circuit import Circuit
    from pydsp_local.modules import *
    from pydsp_local.io_utility import *
    from pydsp_local.file_writers import VHDLWriter, GraphvizWriter
    from .result_subwindow import PlotWindow, DetailsSubWindow, VHDLViewerWindow, DotViewerWindow


except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pydsp_local.circuit import Circuit
    from pydsp_local.modules import *
    from pydsp_local.io_utility import *
    from pydsp_local.file_writers import VHDLWriter, GraphvizWriter
    from result_subwindow import PlotWindow , DetailsSubWindow, VHDLViewerWindow, DotViewerWindow


class PydspHandler():
    def __init__(self):
       pass

    def logger(self, logger_instance):
        return logger_instance

    def create_pydsp_circuit(self, result_model, validate_flag = False, dot_flag = False, wordlength = None):
        # Unpack the dictionary values into the respective variables
        print("res ",result_model)
        h_res = result_model['h_res']
        gain = result_model['gain']
        h = result_model['h']
        alpha = result_model['alpha']
        beta = result_model['beta']
        gamma = result_model['gamma']
        delta = result_model['delta']
        epsilon = result_model['epsilon']
        zeta = result_model['zeta']
        phi = result_model['phi']
        theta = result_model['theta']
        available_dsp = result_model['avail_dsp']
        dsp_values = result_model['dsp_values']
        rho = result_model['rho']
        filter_type = result_model['filter_type']   
        
        # Ensure that the lengths are the same
        adder_count = [len(alpha), len(beta), len(gamma), len(delta), len(epsilon), len(zeta), len(theta)-2-available_dsp]
        # Check if all lengths are the same
        if len(set(adder_count)) != 1:
            raise ValueError(f"Adder_counts inconsistent detected lengths: {adder_count}")
            return
        else:
            print(f"Adder_counts have consistent length: {adder_count[0]}")
            adder_multiplier_length = adder_count[0]

        print(f"adder_multiplier_length{adder_multiplier_length}")
        half_order_count = [len(phi), len(h_res)]

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
        half_order = result_model['half_order']
        wordlength = wordlength if wordlength is not None else result_model['wordlength']
        adder_wordlength = wordlength + result_model['adder_wordlength_ext']
        fracw = result_model['fracw']


        #convert dsp_values to integer
        for i, dsp_value in enumerate(dsp_values):
            dsp_values[i] = dsp_value * 2**fracw



        mult_constant = [None for i in range(len(dsp_values))]

        c = [None for i in range(adder_multiplier_length+2+available_dsp)]
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
        neg_res = [None for i in range(half_order)]

        

        # create a circuit for a simple constant multiplication
        circuit = Circuit()
        # define data type used within the circuit => 8 bit signed numbers
        dat = DataType.SIGNED
      
        i_min_wordlength = -(2**(wordlength-1))
        i_max_wordlength = -i_min_wordlength-1
        if validate_flag:
            i_min_adder_wordlength = -(2**(adder_wordlength-1))
            i_max_adder_wordlength = -i_min_adder_wordlength-1
        else:
            i_min_adder_wordlength = i_min_wordlength
            i_max_adder_wordlength = i_max_wordlength

        # create the input for our circuit
        c[0] = circuit.add(Input(input_ranges=[[i_min_adder_wordlength, i_max_adder_wordlength]], data_type=dat, name="c0"))
        c[-1] = circuit.add(Constant(value = 0, data_type=dat, name="c_last_zero"))
        if validate_flag:
            minus = circuit.add(Constant(value = -1, data_type=dat, name="minus"))

        # Create DSPs multiplier
        for i in range(adder_multiplier_length+1, adder_multiplier_length+1+available_dsp):
            value = dsp_values[i-adder_multiplier_length-1]
            mult_constant[i-adder_multiplier_length-1] = circuit.add(Constant(value = value, data_type=dat, name=f"mult_constant{i}"))
            c[i] = circuit.add(Mult(input_ranges=[c[0].output_range(), mult_constant[i-adder_multiplier_length-1].output_range()], data_type=dat, name=f"Mult{i}"))
            circuit.connect(src_module=mult_constant[i-adder_multiplier_length-1], dst_module=c[i], dst_port=0)
            circuit.connect(src_module=c[0], dst_module=c[i], dst_port=1)

        # Create the adder tree
        for i in range(1, adder_multiplier_length+1):
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
                #do an addition
                for j, beta_val in enumerate(beta[i-1]):
                    if beta_val == 1:
                        z[i-1] = circuit.add(Add(input_ranges=[s[i-1].output_range(), c[j].output_range()], data_type=dat, name=f"Add{i}"))
                        circuit.connect(src_module=s[i-1], dst_module=z[i-1], dst_port=0)
                        circuit.connect(src_module=c[j], dst_module=z[i-1], dst_port=1)

            
            c[i] = circuit.add(ShiftRight(input_ranges=[z[i-1].output_range()], shift_length=zeta_shift_value[i-1], data_type=dat, name=f"zeta_shift{i}"))
            circuit.connect(src_module=z[i-1], dst_module=c[i], dst_port=0)
                

        for m in range(half_order):
            if theta_connection[m] == len(c)-1:
                #if h_res zero skip everything, keep h_ext None at that particular index
                continue

            #use connection
            h_ext[m] = circuit.add(ShiftLeft(input_ranges=[c[theta_connection[m]].output_range()], data_type=dat, name=f"phi_Shift{m}", shift_length=phi_shift_value[m]))
            circuit.connect(src_module=c[theta_connection[m]], dst_module=h_ext[m], dst_port=0)

            if rho[m] == 1:
                neg_res[m] = True
            else:
                neg_res[m] = False

            
            if validate_flag:
                if rho[m] == 1:
                    o[m] = circuit.add(Mult(input_ranges=[h_ext[m].output_range(), minus.output_range()], data_type=dat, name=f"Mult_res{m}"))
                    circuit.connect(src_module=h_ext[m], dst_module=o[m], dst_port=0)
                    circuit.connect(src_module=minus, dst_module=o[m], dst_port=1)

                    h_res_output[m] = circuit.add(Output(input_ranges=[o[m].output_range()], data_type=dat, name=f"Y{m}"))
                    circuit.connect(src_module=o[m], dst_module=h_res_output[m], dst_port=0)
                else:
                    h_res_output[m] = circuit.add(Output(input_ranges=[h_ext[m].output_range()], data_type=dat, name=f"Y{m}"))
                    circuit.connect(src_module=h_ext[m], dst_module=h_res_output[m], dst_port=0)

        # circuit.print_info()
        if validate_flag:
            circuit.validate()
            c[0].define_input_data([1])
            for time_step in range(1):
                for i in range(adder_multiplier_length+1+available_dsp):
                    c_val = c[i].get_output(time_step)
                    print(f"c{i} = {c_val}")
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
        

        h_ext_breakdown = FunctionHandler.flip_filter_array(h_ext, filter_type)
        neg_res_breakdown = FunctionHandler.flip_filter_array(neg_res, filter_type, True)
        h_final = [h_val for h_val in h_ext_breakdown if h_val is not None]
        neg_res_final = [neg_val for neg_val in neg_res_breakdown if neg_val is not None]
        reg = [None for i in range(len(h_final)-1)]
        radd = [None for i in range(len(h_final)-1)]

        #A special Case where the first index is negative
        if neg_res_final[0]:
            neg_h_final = circuit.add(Sub(input_ranges=[c[-1].output_range(), h_final[0].output_range()], data_type=dat, name=f"spec_Rsub"))
            circuit.connect(src_module=c[-1], dst_module=neg_h_final, dst_port=0)
            circuit.connect(src_module=h_final[0], dst_module=neg_h_final, dst_port=1)

        for i in range(len(reg)):
            if i == 0:
                if neg_res_final[0]:
                    reg[i] = circuit.add(Register(input_ranges=[neg_h_final.output_range()], data_type=dat, name=f"Reg{i}"))
                    circuit.connect(src_module=neg_h_final, dst_module=reg[i], dst_port=0)
                else:
                    reg[i] = circuit.add(Register(input_ranges=[h_final[i].output_range()], data_type=dat, name=f"Reg{i}"))
                    circuit.connect(src_module=h_final[i], dst_module=reg[i], dst_port=0)
            else:
                reg[i] = circuit.add(Register(input_ranges=[radd[i-1].output_range()], data_type=dat, name=f"Reg{i}"))
                circuit.connect(src_module=radd[i-1], dst_module=reg[i], dst_port=0)

            if neg_res_final[i+1]:
                radd[i] = circuit.add(Sub(input_ranges=[reg[i].output_range(), h_final[i+1].output_range()], data_type=dat, name=f"Rsub{i}"))
                circuit.connect(src_module=reg[i], dst_module=radd[i], dst_port=0)
                circuit.connect(src_module=h_final[i+1], dst_module=radd[i], dst_port=1)
            else:
                radd[i] = circuit.add(Add(input_ranges=[reg[i].output_range(), h_final[i+1].output_range()], data_type=dat, name=f"Radd{i}"))
                circuit.connect(src_module=reg[i], dst_module=radd[i], dst_port=0)
                circuit.connect(src_module=h_final[i+1], dst_module=radd[i], dst_port=1)

        filter_out = circuit.add(Output(input_ranges=[radd[-1].output_range()], data_type=dat, name=f"filter_out"))
        circuit.connect(src_module=radd[-1], dst_module=filter_out, dst_port=0)

        circuit.validate()
        circuit.print_info()

        if dot_flag:
            GraphvizWriter.write("fir_filter.dot", circuit)
        else:
            VHDLWriter.write("fir_filter.vhdl", circuit)
        print(f"wordlength: {wordlength}")
        

   

class FunctionHandler():
    def __init__(self):
        pass

    @staticmethod
    def flip_filter_array(arr, filter_type, neg_flag = False):
        arr_np = np.array(arr, dtype=object)  # Specify dtype as object to allow None
        if filter_type == 0:
            # Even length: reverse the array excluding the first element (so no repeat of 0) and append the original array
            result = np.concatenate((arr_np[::-1], arr_np[1:]))
        elif filter_type == 1:
            # Odd length: reverse the array and append the original array
            result = np.concatenate((arr_np[::-1], arr_np))
            
        elif filter_type == 2:
            arr_np = np.insert(arr_np, 0, None)
            print("inv_arr:", arr_np)

            if neg_flag:
                inv_arr = arr_np[::-1]
                inv_arr = [not(i) if i is not None else i for i in inv_arr]
                result = np.concatenate((inv_arr, arr_np[1:]))
            #insert 0 at the beginning
            else:
                result = np.concatenate((arr_np[::-1], arr_np[1:]))
       
        elif filter_type == 3:
            # Odd length: reverse the array and append the original array
            if neg_flag:
                inv_arr = arr_np[::-1]
                inv_arr = [not(i) if i is not None else i for i in inv_arr]
                result = np.concatenate((inv_arr, arr_np))
            else:
                result = np.concatenate((arr_np[::-1], arr_np))
        else:
            raise ValueError("Invalid filter type")
        return result
    
class JsonUnloader():
    def __init__(self):
        pass

    @staticmethod
    def load_with_lock(filename_id):
        if filename_id == 0:
            filename='problem_description.json'
        elif filename_id == 1:
            filename='result_valid.json'
        elif filename_id == 2:
            filename='result_leak.json'
        else:
            print("Invalid filename_id")
            return
        
        lock_filename = filename + '.lock'
        lock = FileLock(lock_filename)
        loaded_data = None
        with lock:  # Acquire lock before file access
            with open(filename, 'r') as json_file:
                loaded_data = json.load(json_file)
   
        return loaded_data
    
    @staticmethod
    def format_list_no_spaces(lst):
        return '[' + ','.join(
            JsonUnloader.format_list_no_spaces(item) if isinstance(item, list) else str(item)
            for item in lst
        ) + ']'
    
    @staticmethod
    def format_dict_no_spaces(dct):
        items = []
        for key, value in dct.items():
            if isinstance(value, list):
                value_str = JsonUnloader.format_list_no_spaces(value)
            elif isinstance(value, dict):
                value_str = JsonUnloader.format_dict_no_spaces(value)
            else:
                value_str = str(value)
            items.append(f"'{key}':{value_str}")
        return '{' + ','.join(items) + '}'
        
class DynamicTableWidgetResult(QWidget):
    def __init__(self, table_widget, valid_data, app = None):
        super().__init__()
        self.table = table_widget
        self.valid_data = valid_data
        self.last_max_key = -1
        self.app = app  

    def startTimer(self):
        # Create a timer that calls updateTableData every 1 second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTableData)
        self.timer.start(2000)  # 1000 ms = 1 second

    def updateTableData(self):
        if  self.valid_data:
            filename='result_valid.json'
        else:
            filename='result_leak.json'
        
        if not os.path.exists(filename):
            return
        
        lock_filename = filename + '.lock'
        lock = FileLock(lock_filename)
        data = None

        with lock:  # Acquire lock before file access
            with open(filename, 'r') as json_file:
                data = json.load(json_file)

        max_key = max(map(int, data.keys())) if data else -1

        if max_key > self.last_max_key:
            for key in range(self.last_max_key + 1, max_key + 1):
                if str(key) in data:
                    self.addRow(data[str(key)], str(key))
            self.last_max_key = max_key

    def addRow(self, result_data, key):
        # Get the current row count
        rowCount = self.table.rowCount()

        # Insert a new row
        self.table.insertRow(rowCount)

         # Add row number in the first column
        self.table.setItem(rowCount, 0, QTableWidgetItem(str(result_data['problem_id'])))
        self.table.setItem(rowCount, 1, QTableWidgetItem(str(key)))
        self.table.setItem(rowCount, 2, QTableWidgetItem(str(result_data['option'])))

        # Add placeholder data in the second and third columns, converting to strings
        self.table.setItem(rowCount, 3, QTableWidgetItem(str(result_data['total_adder'])))
        self.table.setItem(rowCount, 4, QTableWidgetItem(str(result_data['adder_s'])))

        # Create buttons in the fourth, fifth, and sixth columns
        button1_plot= QPushButton(f'Plot Result')
        button1_plot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 1))
        self.table.setCellWidget(rowCount, 5, button1_plot)

        button2_show = QPushButton(f'Show Details')
        button2_show.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 2))
        self.table.setCellWidget(rowCount, 6, button2_show)

        button3_vhdl = QPushButton(f'Generate VHDL Code')
        button3_vhdl.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 3))
        self.table.setCellWidget(rowCount, 7, button3_vhdl)

        button4_dot = QPushButton(f'Generate DOT data')
        button4_dot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 4))
        self.table.setCellWidget(rowCount, 8, button4_dot)

        for i in range(0, 5):
            self.table.item(rowCount, i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)


    def buttonClicked(self, row, button_number):
        print(f'Button {button_number} clicked for row {row}')
        problem_id = self.table.item(row - 1, 0)
        result_id = self.table.item(row - 1 , 1)

        if button_number == 1:
            self.plot_result(problem_id.text(), result_id.text())
        elif button_number == 2:
            self.show_details(problem_id.text(), result_id.text())
        elif button_number == 3:
            self.generate_vhdl(problem_id.text(), result_id.text())
        elif button_number == 4:
            self.generate_dot(problem_id.text(), result_id.text())

    def generate_dot(self, problem_id, result_id):
        file = 1 if self.valid_data else 2
        data_result = JsonUnloader.load_with_lock(file)[result_id]
        print("data_result", data_result)
        pydsp_handler = PydspHandler()
        flag = pydsp_handler.create_pydsp_circuit(data_result, True)
        
        if flag:
            self.app.logger.plog("Adder graph is validated")
            
        else:
            self.app.logger.plog("Adder graph is Somehow invalid")

        pydsp_handler.create_pydsp_circuit(data_result, False, dot_flag = True)
        self.vhdl_window = DotViewerWindow(dot_file_path="fir_filter.dot")
        self.app.logger.plog("Dot file generated successfully")
        self.vhdl_window.show()

    def plot_result(self, problem_id, result_id):
        file = 1 if self.valid_data else 2
        result_data = JsonUnloader.load_with_lock(file)
        problem_data = JsonUnloader.load_with_lock(0)

        if result_id in result_data:
            print(f"Plotting result for problem {problem_id}, result {result_id}")
        else:
            print(f"Result id {result_id} not found in data")
            return
        
        if problem_id in problem_data:
            print(f"Plotting result for problem {problem_id}, result {result_id}")
        else:  
            print(f"Problem {problem_id} not found in data")
            return
        
        # Create a new plot window
        self.plot_window = PlotWindow(self.app.day_night, problem_data[problem_id], result_data[result_id])
        self.plot_window.show()

    def generate_vhdl(self, problem_id, result_id):
        file = 1 if self.valid_data else 2
        data_result = JsonUnloader.load_with_lock(file)[result_id]
       
        pydsp_handler = PydspHandler()
        flag = pydsp_handler.create_pydsp_circuit(data_result, True)

        if flag:
            self.app.logger.plog("Adder graph is validated")
            
        else:
            self.app.logger.plog("Adder graph is Somehow invalid")

        if self.app.vhdl_input_word_box.value() < 2:
            self.app.show_error_dialog("Wordlength must be at least 2, change it in the result settings -> VHDL wordlength")
            return

        pydsp_handler.create_pydsp_circuit(data_result, False, wordlength = self.app.vhdl_input_word_box.value())
        self.vhdl_window = VHDLViewerWindow()
        self.app.logger.plog("VHDL code generated successfully")
        self.vhdl_window.show()
        

    def show_details(self, problem_id, result_id):
        file = 1 if self.valid_data else 2
        data_problem = JsonUnloader.load_with_lock(0)[problem_id]
        data_result = JsonUnloader.load_with_lock(file)[result_id]

        data_problem.pop("as_results", None)
        data_problem.pop("as_target_result", None)
        data_problem.pop("deepsearch_am_cost", None)
        data_problem.pop("deepsearch_h_zero", None)
        data_problem.pop("deepsearch_target_result", None)
        data_problem.pop("cutoffs_lower_ydata_lin", None)
        data_problem.pop("cutoffs_upper_ydata_lin", None)
        data_problem.pop("cutoffs_x", None)
        data_problem.pop("presolve_result", None)
        data_problem.pop("original_xdata", None)
        data_problem.pop("original_upperbound_lin", None)
        data_problem.pop("original_lowerbound_lin", None)
        data_problem.pop("continue_solver", None)
        data_problem.pop("gain_intW", None)
        data_problem.pop("gain_lowerbound", None)
        data_problem.pop("gain_upperbound", None)
        data_problem.pop("gain_wordlength", None)
        data_problem.pop("gurobi_auto_thread", None)
        data_problem.pop("ignore_lowerbound", None)
        data_problem.pop("adder_count", None)


        data_result.pop("adder_depth", None)
        data_result.pop("am_count", None)
        data_result.pop("adder_wordlength", None)
        

        print(f"Showing details for problem {problem_id}, result {result_id}")
        self.detail_window = DetailsSubWindow(data_problem, data_result)
        self.detail_window.show()

class DynamicTableWidgetProblems(QWidget):
    def __init__(self, table_widget, app = None):
        super().__init__()
        self.table = table_widget
        self.last_max_key = -1
        self.app = app  

    def startTimer(self):
        # Create a timer that calls updateTableData every 1 second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTableData)
        self.timer.start(2000)  # 1000 ms = 1 second

    def updateTableData(self):
        filename='problem_description.json'
        
        if not os.path.exists(filename):
            return
        
        lock_filename = filename + '.lock'
        lock = FileLock(lock_filename)
        data = None

        with lock:  # Acquire lock before file access
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
    

        max_key = max(map(int, data.keys())) if data else -1

        if max_key > self.last_max_key:
            for key in range(self.last_max_key + 1, max_key + 1):
                if str(key) in data:
                    self.addRow(data[str(key)], str(key))
            self.last_max_key = max_key

        for key, value in data.items():
            self.update_table(value, key)
    

    def update_table(self, result_data, key):
        rowCount = int(key)

        # Add row number in the first column
        self.table.setItem(rowCount, 0, QTableWidgetItem(str(key)))

        # Add placeholder data in the second and third columns, converting to strings
        done = "done" if result_data['done'] == True else "Not Done"
        self.table.setItem(rowCount, 1, QTableWidgetItem(done))

        solution = "Not Found"
        if "as_target_result" in result_data:
            solution = "AS Search" if result_data["as_target_result"] else solution
        if "deepsearch_target_result" in result_data:
            solution = "deepsearch" if result_data["deepsearch_target_result"] else solution
                           

        self.table.setItem(rowCount, 2, QTableWidgetItem(solution))
        for i in range(0, 3):
            self.table.item(rowCount, i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)


    def addRow(self, result_data, key):
        # Get the current row count
        rowCount = self.table.rowCount()

        # Insert a new row
        self.table.insertRow(rowCount)

        # Add row number in the first column
        self.table.setItem(rowCount, 0, QTableWidgetItem(str(key)))

        # Add placeholder data in the second and third columns, converting to strings
        done = "done" if result_data['done'] == True else "Not Done"
        self.table.setItem(rowCount, 1, QTableWidgetItem(done))

        solution = "Not Found"
        if "as_target_result" in result_data:
            solution = "AS Search" if result_data["as_target_result"] else solution
        if "deepsearch_target_result" in result_data:
            solution = "deepsearch" if result_data["deepsearch_target_result"] else solution
                           

        self.table.setItem(rowCount, 2, QTableWidgetItem(solution))

        button1_plot= QPushButton(f'Continue Solver')
        button1_plot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 1))
        self.table.setCellWidget(rowCount, 3, button1_plot)

        # Create buttons in the fourth, fifth, and sixth columns
        button1_plot= QPushButton(f'Plot Result')
        button1_plot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 2))
        self.table.setCellWidget(rowCount, 4, button1_plot)

        button2_show = QPushButton(f'Show Details')
        button2_show.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 3))
        self.table.setCellWidget(rowCount, 5, button2_show)

        button3_vhdl = QPushButton(f'Generate VHDL Code')
        button3_vhdl.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 4))
        self.table.setCellWidget(rowCount, 6, button3_vhdl)

        button4_dot = QPushButton(f'Generate DOT data')
        button4_dot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 5))
        self.table.setCellWidget(rowCount, 7, button4_dot)

        for i in range(0, 3):
            self.table.item(rowCount, i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def buttonClicked(self, row, button_number):
        print(f'Button {button_number} clicked for row {row}')
        problem_id = self.table.item(row - 1, 0)
        problem_data = JsonUnloader.load_with_lock(0)
        try:
            problem_data = problem_data[problem_id.text()]
        except KeyError:
            print(f"Problem {problem_id.text()} not found in data")
            return
        print(f"Problem {problem_id.text()} found in data")

        if button_number == 1:
            self.continue_solver(problem_data, problem_id.text())
        elif button_number == 2:
            self.plot_result(problem_data)
        elif button_number == 3:
            self.show_details(problem_data)
        elif button_number == 4:
            self.generate_vhdl(problem_data)
        elif button_number == 5:
            self.generate_dot(problem_data)
    

    def generate_dot(self, problem_data):
        data_problem = problem_data.copy()
        data_result = None

        if "as_results" in data_problem:
            data_result = problem_data['as_target_result'] if problem_data['as_target_result'] else None
        if "deepsearch_target_result" in data_problem:
            data_result = problem_data['deepsearch_target_result'] if problem_data['deepsearch_target_result'] else data_result

        if not(data_problem) or not(data_result):
            print("No data found")
            return
        data_result.update({
            'wordlength': data_problem['wordlength'],
            'adder_wordlength': data_problem['adder_wordlength_ext']+data_problem['wordlength'],
            'fracw': data_problem['wordlength']-data_problem['intW'],
        })
        pydsp_handler = PydspHandler()
        flag = pydsp_handler.create_pydsp_circuit(data_result, True)
        
        if flag:
            self.app.logger.plog("Adder graph is validated")
            
        else:
            self.app.logger.plog("Adder graph is Somehow invalid")

        pydsp_handler.create_pydsp_circuit(data_result, False, dot_flag = True)
        self.vhdl_window = DotViewerWindow(dot_file_path="fir_filter.dot")
        self.app.logger.plog("Dot file generated successfully")
        self.vhdl_window.show()


    def continue_solver(self, problem_data,problem_id):
        if "done" in problem_data:
            if problem_data["done"] != "False":
                self.app.logger.plog("Problem already solved")
                return
        
        input_data = {
        'filter_type': problem_data['filter_type'],
        'order_upperbound': problem_data['order_upperbound'],
        'original_xdata': problem_data['original_xdata'],
        'original_upperbound_lin': problem_data['original_upperbound_lin'],
        'original_lowerbound_lin': problem_data['original_lowerbound_lin'],
        'cutoffs_x': problem_data['cutoffs_x'],
        'cutoffs_upper_ydata_lin': problem_data['cutoffs_upper_ydata_lin'],
        'cutoffs_lower_ydata_lin': problem_data['cutoffs_lower_ydata_lin'],
        'ignore_lowerbound': problem_data['ignore_lowerbound'],
        'wordlength': problem_data['wordlength'],
        'adder_depth': problem_data['adder_depth'],
        'avail_dsp': problem_data['avail_dsp'],
        'adder_wordlength_ext': problem_data['adder_wordlength_ext'],
        'gain_upperbound': problem_data['gain_upperbound'],
        'gain_lowerbound': problem_data['gain_lowerbound'],
        'coef_accuracy': problem_data['coef_accuracy'],
        'intW': problem_data['intW'],
        'gurobi_thread': problem_data['gurobi_thread'],
        'pysat_thread': problem_data['pysat_thread'],
        'z3_thread': problem_data['z3_thread'],
        'timeout': problem_data['timeout'],
        'start_with_error_prediction': problem_data['start_with_error_prediction'],
        'solver_accuracy_multiplier': problem_data['solver_accuracy_multiplier'],
        'start_with_error_prediction': problem_data['start_with_error_prediction'],
        'deepsearch': problem_data['deepsearch'],
        'patch_multiplier' : problem_data['patch_multiplier'],
        'gurobi_auto_thread': problem_data['gurobi_auto_thread'],
        'worker': problem_data['worker'],
        'search_step':problem_data['search_step'],
        'continue_solver': True,
        'problem_id': problem_id,
        }
        
        self.app.continue_solver(input_data)

    def plot_result(self, problem_data):
        data_problem = problem_data.copy()
        data_result = None
        if "as_results" in data_problem:
            data_result = problem_data['as_target_result'] if problem_data['as_target_result'] else None
        if "deepsearch_target_result" in data_problem:
            data_result = problem_data['deepsearch_target_result'] if problem_data['deepsearch_target_result'] else data_result

        if not(data_problem) or not(data_result):
            print("No data found")
            return
        
        # Create a new plot window
        self.plot_window = PlotWindow(self.app.day_night, problem_data, data_result)
        self.plot_window.show()

    def generate_vhdl(self, problem_data):
        data_problem = problem_data.copy()
        data_result = None
        if "as_results" in data_problem:
            data_result = problem_data['as_target_result'] if problem_data['as_target_result'] else None
        if "deepsearch_target_result" in data_problem:
            data_result = problem_data['deepsearch_target_result'] if problem_data['deepsearch_target_result'] else data_result

        if not(data_problem) or not(data_result):
            print("No data found")
            return
        print("data_problem", data_problem)
        data_result.update({
            'wordlength': data_problem['wordlength'],
            'adder_wordlength': data_problem['adder_wordlength_ext']+data_problem['wordlength'],
            'fracw': data_problem['wordlength']-data_problem['intW'],
        })

        
        pydsp_handler = PydspHandler()
        flag = pydsp_handler.create_pydsp_circuit(data_result, True)
        if flag:
            self.app.logger.plog("Adder graph is validated")
            pydsp_handler.create_pydsp_circuit(data_result, False, wordlength = self.app.vhdl_input_word_box.value())
            self.vhdl_window = VHDLViewerWindow()
            self.app.logger.plog("VHDL code generated successfully")

            self.vhdl_window.show()

        

    def show_details(self, problem_data):
        data_problem = problem_data
        data_result = None
        if "as_results" in data_problem:
            data_result = problem_data['as_target_result'] if problem_data['as_target_result'] else None
        if "deepsearch_target_result" in data_problem:
            data_result = problem_data['deepsearch_target_result'] if problem_data['deepsearch_target_result'] else data_result

        if not(data_problem):
            print("No data found")
            return
        
        data_problem.pop("as_results", None)
        data_problem.pop("as_target_result", None)
        data_problem.pop("deepsearch_am_cost", None)
        data_problem.pop("deepsearch_h_zero", None)
        data_problem.pop("deepsearch_target_result", None)
        data_problem.pop("cutoffs_lower_ydata_lin", None)
        data_problem.pop("cutoffs_upper_ydata_lin", None)
        data_problem.pop("cutoffs_x", None)
        data_problem.pop("presolve_result", None)
        data_problem.pop("original_xdata", None)
        data_problem.pop("original_upperbound_lin", None)
        data_problem.pop("original_lowerbound_lin", None)
        data_problem.pop("continue_solver", None)
        data_problem.pop("gain_intW", None)
        data_problem.pop("gain_wordlength", None)
        data_problem.pop("ignore_lowerbound", None)
        data_problem.pop("adder_count", None)

        if data_result:
            data_result.pop("adder_wordlength", None)
        
        self.detail_window = DetailsSubWindow(data_problem, data_result)
        self.detail_window.show()




if __name__ == '__main__':

    input_data = {
        'input_wordlength' : 9
    }

    result = JsonUnloader()
    loaded_data = result.load_with_lock( True)
    print(loaded_data["2"])


    # for key_subitem, value_subitem in loaded_data['0'].items():
    #         print(f"Key:{key_subitem}, Value: {value_subitem}")
    
    pydsp = PydspHandler()
    pydsp.create_pydsp_circuit(loaded_data['2'], True)
    pydsp.create_pydsp_circuit(loaded_data['2'], False)

    # for key, value in loaded_data.items():
    #     for key_subitem, value_subitem in loaded_data[key].items():
    #         print(f"Key:{key}, Value: {value_subitem}")

    
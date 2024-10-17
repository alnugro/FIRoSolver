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
    from pydsp_local.file_writers import VHDLWriter
    from .result_subwindow import PlotWindow


except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pydsp_local.circuit import Circuit
    from pydsp_local.modules import *
    from pydsp_local.io_utility import *
    from pydsp_local.file_writers import VHDLWriter
    from result_subwindow import PlotWindow 


class PydspHandler():
    def __init__(self):
       pass

    def logger(self, logger_instance):
        return logger_instance

    def create_pydsp_circuit(self, result_model, validate_flag = False):
        # Unpack the dictionary values into the respective variables
        h_res = result_model['h_res']
        self.gain = result_model['gain']
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
            c[0].define_input_data([1])
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
        

        h_ext_breakdown = self.flip_filter_array(h_ext)
        h_final = [h_val for h_val in h_ext_breakdown if h_val is not None]
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
        
class DynamicTableWidget(QWidget):
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

        # Add placeholder data in the second and third columns, converting to strings
        self.table.setItem(rowCount, 2, QTableWidgetItem(str(result_data['total_adder'])))
        self.table.setItem(rowCount, 3, QTableWidgetItem(str(result_data['adder_s'])))

        # Create buttons in the fourth, fifth, and sixth columns
        button1_plot= QPushButton(f'Plot Result')
        button1_plot.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 1))
        self.table.setCellWidget(rowCount, 4, button1_plot)

        button2_show = QPushButton(f'Show Details')
        button2_show.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 2))
        self.table.setCellWidget(rowCount, 5, button2_show)

        button3_vhdl = QPushButton(f'Generate VHDL Code')
        button3_vhdl.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 3))
        self.table.setCellWidget(rowCount, 6, button3_vhdl)

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
        data = JsonUnloader.load_with_lock(file, self.valid_data)

        if result_id in data:
            print(f"Plotting result for problem {problem_id}, result {result_id}")
        else:
            print(f"Result id {result_id} not found in data")
            return

    def show_details(self, problem_id, result_id):
        data = JsonUnloader.load_with_lock(0, self.valid_data)

        if problem_id in data:
            print(f"Showing details for problem {problem_id}, result {result_id}") 
        else:
            print(f"Problem {problem_id} not found in data")
            return




if __name__ == '__main__':

    input_data = {
        'input_wordlength' : 9
    }

    result = JsonUnloader()
    loaded_data = result.load_with_lock( True)

    for key_subitem, value_subitem in loaded_data['0'].items():
            print(f"Key:{key_subitem}, Value: {value_subitem}")
    
    pydsp = PydspHandler()
    pydsp.create_pydsp_circuit(loaded_data['0'], True)
    pydsp.create_pydsp_circuit(loaded_data['0'], False)

    # for key, value in loaded_data.items():
    #     for key_subitem, value_subitem in loaded_data[key].items():
    #         print(f"Key:{key}, Value: {value_subitem}")

    
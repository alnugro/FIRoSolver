import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLabel, QMessageBox, QHBoxLayout, QSpinBox, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, pyqtSignal
import json
from filelock import FileLock
import pprint
import os
import webbrowser
import urllib.parse

class PlotWindow(QWidget):
    def __init__(self, day_night, problem_data, result_data):
        super().__init__()
        self.setWindowTitle("Result Plot")
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
        self.setGeometry(400, 400, 1200, 800)
        self.day_night = day_night
        self.original_xdata = problem_data['original_xdata']
        self.filter_type = problem_data['filter_type']
        self.order_upperbound = problem_data['order_upperbound']
        self.original_lowerbound_lin = problem_data['original_lowerbound_lin']
        self.original_upperbound_lin = problem_data['original_upperbound_lin']
        self.ignore_lowerbound = problem_data['ignore_lowerbound']

        self.h_res = result_data['h_res']
        self.gain = result_data['gain']

        self.half_order = (self.order_upperbound // 2) if self.filter_type == 0 or self.filter_type == 2 else (self.order_upperbound // 2) - 1

        # Create a layout for the window
        layout = QVBoxLayout()

        # Create a Matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        # Create a toolbar for zoom, pan, save, etc.
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add the toolbar and canvas to the layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.plot_graph()

    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            cm=(2*np.cos(np.pi*omega*m))
            return cm
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))
        
    def plot_graph(self):
        if self.day_night:
            plt.style.use('fivethirtyeight')
        else:
            plt.style.use('dark_background')
        
        ax = self.figure.add_subplot(111)


        magnitude_response = []
        
        # Recompute the frequency response for each frequency point
        for i, omega in enumerate(self.original_xdata):
            term_sum_exprs = 0
            # Compute the sum of products of coefficients and the cosine/sine terms with much higher cm accuracy
            for j in range(self.half_order+1):
                cm_const = self.cm_handler(j, omega)
                term_sum_exprs += self.h_res[j] * cm_const
            
            # Append the computed sum expression to the frequency response list
            magnitude_response.append(np.abs(term_sum_exprs))

        leaks = []
        leaks_mag = []
        continous_leak_count = 0
        continous_flag = False

        # Check for leaks by comparing the FFT result with the 10x accuracy bounds
        for i, mag in enumerate(magnitude_response):
            if mag > self.original_upperbound_lin[i] * self.gain: 
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append((mag-self.original_upperbound_lin[i])/self.gain)
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            elif mag < self.original_lowerbound_lin[i] * self.gain:
                if mag < self.ignore_lowerbound:
                    continue
                leaks.append((i, mag))  # Collect the leak points
                leaks_mag.append((mag-self.original_lowerbound_lin[i])/self.gain)
                if continous_flag == False:
                    continous_leak_count +=1
                continous_flag = True
            else: continous_flag = False

      
        # Plot the input bounds (using the original bounds, which are at higher accuracy)
        ax.scatter(self.original_xdata, np.array(self.original_upperbound_lin) * self.gain, color='r', s=20, picker=5, label="Upper Bound")
        ax.scatter(self.original_xdata, np.array(self.original_lowerbound_lin) * self.gain, color='b', s=20, picker=5, label="Lower Bound")

        # Plot the magnitude response from the calculated coefficients
        ax.scatter(self.original_xdata, magnitude_response, color='y', label="Magnitude Response", s=10, picker=5)

        # Mark the leaks on the plot
        if leaks:
            leak_indices, leak_values = zip(*leaks)
            leaks_mag = [float(x) for x in leaks_mag]


            leak_freqs = [self.original_xdata[i] for i in leak_indices]
            ax.scatter(leak_freqs, leak_values, color='cyan', s=4, label="Leak Points", zorder=5)

        
        
        # Draw the plot
        self.canvas.draw()


class DetailsSubWindow(QWidget):
    def __init__(self, data_problem, data_result):
        super().__init__()
        h_res_int = []
        for i in range(len(data_result['h_res'])):
            h_res_int.append(data_result['h_res'][i] * 2**data_result['fracw'])
        data_result.update({'h_res_int': h_res_int})

        self.setWindowTitle("Problem and Result Details")
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
        self.setGeometry(400, 400, 800, 600)

        # Create a main layout
        main_layout = QVBoxLayout()

        # Result Data Section
        result_label = QLabel("Result Data")
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        result_text_edit = QTextEdit()
        result_text_edit.setReadOnly(True)
        result_text_edit.setFont(QFont("Courier", 10))
        if data_result is None:
            result_text_edit.setText("No result data available.")
        else:
            result_text_edit.setText(self.format_result_data(data_result))

        # Problem Data Section
        problem_label = QLabel("Problem Description")
        problem_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        problem_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        problem_text_edit = QTextEdit()
        problem_text_edit.setReadOnly(True)
        problem_text_edit.setFont(QFont("Courier", 10))
        problem_text_edit.setText(self.format_problem_data(data_problem))

        # Add widgets to the main layout
        main_layout.addWidget(result_label)
        main_layout.addWidget(result_text_edit)
        main_layout.addWidget(problem_label)
        main_layout.addWidget(problem_text_edit)

        self.setLayout(main_layout)

    def format_result_data(self, data_result):
        # Reorder the keys to prioritize h_res, adder_m, and adder_s
        keys_order = ['problem_id','solver','h_res', 'adder_m', 'adder_s','total_adder'] + [k for k in data_result.keys() if k not in ['h_res', 'adder_m', 'adder_s']]
        lines = []
        for key in keys_order:
            if key in data_result:
                value = data_result[key]
                formatted_value = self.format_value(value)
                formatted_value_lines = formatted_value.split('\n')
                # The first line with key and colon
                line = f"{key:<25}: {formatted_value_lines[0]}"
                lines.append(line)
                # The subsequent lines, we need to indent appropriately
                for cont_line in formatted_value_lines[1:]:
                    lines.append('\t' + cont_line)
        return "\n".join(lines)

    def format_problem_data(self, data_problem):
        lines = []
        for key, value in data_problem.items():
            formatted_value = self.format_value(value)
            formatted_value_lines = formatted_value.split('\n')
            # The first line with key and colon
            line = f"{key:<25}: {formatted_value_lines[0]}"
            lines.append(line)
            # The subsequent lines, we need to indent appropriately
            for cont_line in formatted_value_lines[1:]:
                lines.append('\t' + cont_line)
        return "\n".join(lines)

    def format_value(self, value, indent=0):
        if isinstance(value, (list, dict)):
            formatted_value = pprint.pformat(value, width=60, indent=0)
            # Remove spaces after '[' or '{' in the first line
            lines = formatted_value.split('\n')
            if lines[0].startswith('[') or lines[0].startswith('{'):
                # Remove any spaces after the opening bracket
                lines[0] = lines[0][0] + lines[0][1:].lstrip()
            # Adjust indentation
            indented_lines = [lines[0]]
            for line in lines[1:]:
                indented_lines.append('\t       ' * (indent + 1) + line)
            return '\n'.join(indented_lines)
        else:
            return str(value)
        

class VHDLViewerWindow(QWidget):
    def __init__(self, vhdl_file_path='fir_filter.vhdl'):
        super().__init__()
        self.setWindowTitle("VHDL Viewer")
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
        self.setGeometry(400, 400, 800, 600)

        self.vhdl_file_path = vhdl_file_path

        # Create the main layout
        main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Generated VHDL")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        main_layout.addWidget(title_label)

        # Text Edit to display the VHDL content
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))

        # Load and display the VHDL file content
        self.load_vhdl_content()

        main_layout.addWidget(self.text_edit)

        # Copy to Clipboard Button
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        main_layout.addWidget(copy_button)

        self.setLayout(main_layout)

    def load_vhdl_content(self):
        # Check if the file exists
        if not os.path.exists(self.vhdl_file_path):
            QMessageBox.critical(self, "Error", f"File not found: {self.vhdl_file_path}")
            return

        # Read the VHDL file content
        try:
            with open(self.vhdl_file_path, 'r', encoding='utf-8') as file:
                vhdl_content = file.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read the file:\n{e}")
            return

        # Set the content to the text edit
        self.text_edit.setPlainText(vhdl_content)

    def copy_to_clipboard(self):
        # Get the VHDL content
        vhdl_content = self.text_edit.toPlainText()
        if vhdl_content:
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(vhdl_content)
        else:
            QMessageBox.warning(self, "Warning", "No content to copy.")





class DotViewerWindow(QWidget):
    def __init__(self, dot_file_path='example.dot'):
        super().__init__()
        self.setWindowTitle("Dot File Viewer")
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
        self.setGeometry(400, 400, 800, 600)

        self.dot_file_path = dot_file_path

        # Create the main layout
        main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Dot File Content")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        main_layout.addWidget(title_label)

        # Text Edit to display the dot file content
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))

        # Load and display the dot file content
        self.load_dot_content()

        main_layout.addWidget(self.text_edit)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Copy to Clipboard Button
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_button)

        # Hyperlink Button to open the URL
        link_button = QPushButton("Open in Graphviz Online")
        link_button.clicked.connect(self.open_graphviz_online)
        button_layout.addWidget(link_button)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_dot_content(self):
        # Check if the file exists
        if not os.path.exists(self.dot_file_path):
            QMessageBox.critical(self, "Error", f"File not found: {self.dot_file_path}")
            return

        # Read the dot file content
        try:
            with open(self.dot_file_path, 'r', encoding='utf-8') as file:
                dot_content = file.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read the file:\n{e}")
            return

        # Set the content to the text edit
        self.text_edit.setPlainText(dot_content)

    def copy_to_clipboard(self):
        # Get the dot file content
        dot_content = self.text_edit.toPlainText()
        if dot_content:
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(dot_content)
        else:
            QMessageBox.warning(self, "Warning", "No content to copy.")

    def open_graphviz_online(self):
        # Get the dot file content
        dot_content = self.text_edit.toPlainText()
        if not dot_content:
            QMessageBox.warning(self, "Warning", "No content to open.")
            return

        # URL-encode the DOT data
        encoded_dot = urllib.parse.quote(dot_content)

        # Construct the URL with the encoded DOT data
        base_url = 'https://dreampuf.github.io/GraphvizOnline/#'
        full_url = base_url + encoded_dot

        # Open the URL in the default web browser
        webbrowser.open(full_url)


class AutomaticParameterResultSubWindow(QWidget):
    continue_signal = pyqtSignal(int, int)  # filter_order, word_length
    cancel_signal = pyqtSignal()

    def __init__(self, data_dict):
        super().__init__()
        self.setWindowTitle("Automatic parameter result")
        self.setWindowIcon(QIcon("GUI/icon/icon.png"))
        self.setGeometry(400, 400, 800, 600)

        # Create main layout
        main_layout = QVBoxLayout()

        # Top section with QSpinBoxes and labels
        spinbox_layout = QVBoxLayout()
        filter_order_layout = QHBoxLayout()
        word_length_layout = QHBoxLayout()

        top_label = QLabel("Found Parameters")
        top_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        # Filter Type Order SpinBox and Label
        filter_order_label = QLabel("Filter Type Order        ")
        self.filter_order_spinbox = QSpinBox()
        self.filter_order_spinbox.setMinimum(1)
        self.filter_order_spinbox.setMaximum(100)
        self.filter_order_spinbox.setValue(data_dict.get('filter_order', 1))
        self.filter_order_spinbox.setSingleStep(1)
        self.filter_order_spinbox.setFixedSize(300, 40)  # Width: 100, Height: 50


        # Word Length SpinBox and Label
        word_length_label = QLabel("Word Length              ")
        self.word_length_spinbox = QSpinBox()
        self.word_length_spinbox.setMinimum(1)
        self.word_length_spinbox.setMaximum(64)
        self.word_length_spinbox.setValue(data_dict.get('wordlength', 16))
        self.word_length_spinbox.setSingleStep(1)
        self.word_length_spinbox.setFixedSize(300, 40)  # Width: 100, Height: 50

        # Add widgets to filter order layout
        filter_order_layout.addWidget(filter_order_label)
        filter_order_layout.addWidget(self.filter_order_spinbox)
        filter_order_layout.addStretch()  # Add stretch before the buttons

        # Add widgets to word length layout
        word_length_layout.addWidget(word_length_label)
        word_length_layout.addWidget(self.word_length_spinbox)
        word_length_layout.addStretch() 

        # Add spinbox sections to spinbox layout
        spinbox_layout.addLayout(filter_order_layout)
        spinbox_layout.addLayout(word_length_layout)

        # QTextEdit with dict as its input
        dict_label = QLabel("Found Results")
        dict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dict_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        dict_text_edit = QTextEdit()
        dict_text_edit.setReadOnly(True)
        dict_text_edit.setFont(QFont("Courier", 10))
        dict_text_edit.setText(self.format_dict_data(data_dict))
        dict_text_edit.setFixedSize(800, 200)

        # Bottom label
        bottom_label = QLabel("Continue the solver with the given parameters?")
        bottom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_label.setStyleSheet("font-size: 16px;")

       # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # Add stretch before the buttons
        continue_button = QPushButton("Continue")
        continue_button.setFixedSize(200, 40)
        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedSize(200, 40)
        button_layout.addWidget(continue_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()  # Add stretch after the buttons

        # Add widgets to main layout
        main_layout.addWidget(top_label)
        main_layout.addLayout(spinbox_layout)
        main_layout.addWidget(dict_label)
        main_layout.addWidget(dict_text_edit)
        main_layout.addWidget(bottom_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Connect buttons
        continue_button.clicked.connect(self.continue_clicked)
        cancel_button.clicked.connect(self.cancel_clicked)


    def format_dict_data(self, data_dict):
        lines = []
        for key, value in data_dict.items():
            formatted_value = self.format_value(value)
            formatted_value_lines = formatted_value.split('\n')
            # First line with key and colon
            line = f"{key:<25}: {formatted_value_lines[0]}"
            lines.append(line)
            # Indent subsequent lines
            for cont_line in formatted_value_lines[1:]:
                lines.append('\t' + cont_line)
        return "\n".join(lines)
    
    def format_value(self, value, indent=0):
        if isinstance(value, (list, dict)):
            formatted_value = pprint.pformat(value, width=60, indent=0)
            lines = formatted_value.split('\n')
            if lines[0].startswith('[') or lines[0].startswith('{'):
                lines[0] = lines[0][0] + lines[0][1:].lstrip()
            indented_lines = [lines[0]]
            for line in lines[1:]:
                indented_lines.append('\t       ' * (indent + 1) + line)
            return '\n'.join(indented_lines)
        else:
            return str(value)
    
    def continue_clicked(self):
        # Retrieve values from spinboxes
        filter_order = self.filter_order_spinbox.value()
        word_length = self.word_length_spinbox.value()
        # Implement the continuation logic here
        self.continue_signal.emit(filter_order, word_length)
        self.close()
    
    def cancel_clicked(self):
        self.cancel_signal.emit()
        self.close()



# Example usage
if __name__ == "__main__":
    # Function to call the subwindow
    def show_automatic_parameter_result_subwindow(data_dict):
        app = QApplication(sys.argv)
        window = AutomaticParameterResultSubWindow(data_dict)
        window.show()
        sys.exit(app.exec())

    data_dict = {
        'filter_order': 5,
        'word_length': 16,
        'other_data': [1, 2, 3],
        'more_info': {'key1': 'value1', 'key2': 'value2'}
    }
    show_automatic_parameter_result_subwindow(data_dict)
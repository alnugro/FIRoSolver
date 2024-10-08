import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotWindow(QWidget):
    def __init__(self, day_night, problem_data, result_data):
        super().__init__()
        self.setWindowTitle("Result Plot")
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
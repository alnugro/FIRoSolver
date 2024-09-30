import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


# New Window with Matplotlib Plot
class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 600, 400)
        
        # Create a layout
        layout = QVBoxLayout()
        
        # Create a Matplotlib figure and add it to the layout
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        # Plot something
        self.plot()

    def plot(self):
        ax = self.canvas.figure.add_subplot(111)
        
        # Example plot: A simple sine wave
        t = np.linspace(0, 10, 500)
        y = np.sin(t)
        
        ax.plot(t, y)
        ax.set_title("Sine Wave")
        
        # Draw the canvas to show the plot
        self.canvas.draw()


# Main Window Class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 300)
        
        # Button to open the new window
        self.button = QPushButton("Open Plot Window", self)
        self.button.clicked.connect(self.open_plot_window)
        
        self.setCentralWidget(self.button)
        
        # Hold a reference to the plot window
        self.plot_window = None

    # Method to open the plot window
    def open_plot_window(self):
        if self.plot_window is None:
            self.plot_window = PlotWindow()  # Create new plot window instance
        self.plot_window.show()  # Show the plot window


# Application setup
app = QApplication(sys.argv)

main_window = MainWindow()
main_window.show()

sys.exit(app.exec())

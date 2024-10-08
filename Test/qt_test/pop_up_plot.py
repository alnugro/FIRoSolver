import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Window")
        self.setGeometry(400, 400, 1200, 800)

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

    def plot_graph(self):
        # Generate some data to plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # Add the plot to the figure
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, label="sin(x)")
        ax.set_title("Sine Wave")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()
        
        # Draw the plot
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 300)

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()

        # Create a button that opens the plot window
        self.button = QPushButton("Open Plot Window")
        self.button.clicked.connect(self.open_plot_window)
        layout.addWidget(self.button)
        
        central_widget.setLayout(layout)

    def open_plot_window(self):
        # Create and show the plot window
        self.plot_window = PlotWindow()
        self.plot_window.show()

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec())

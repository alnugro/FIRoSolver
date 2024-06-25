import sys
import random
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Matplotlib with PyQt6 Example')
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        main_layout = QHBoxLayout(self.main_widget)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.toolbar)

        self.plot_button = QPushButton("Plot Random Data")
        self.plot_button.clicked.connect(self.plot_data)
        button_layout.addWidget(self.plot_button)

        self.zoom_button = QPushButton("Zoom")
        self.zoom_button.clicked.connect(self.zoom)
        button_layout.addWidget(self.zoom_button)

        self.pan_button = QPushButton("Pan")
        self.pan_button.clicked.connect(self.pan)
        button_layout.addWidget(self.pan_button)

        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(self.home)
        button_layout.addWidget(self.home_button)

        main_layout.addLayout(plot_layout)
        main_layout.addLayout(button_layout)

    def plot_data(self):
        self.ax.clear()
        data = [random.random() for _ in range(25)]
        self.ax.plot(data, '*-')
        self.canvas.draw()

    def zoom(self):
        self.toolbar.zoom()

    def pan(self):
        self.toolbar.pan()

    def home(self):
        self.toolbar.home()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = PlotWindow()
    main.show()
    sys.exit(app.exec())

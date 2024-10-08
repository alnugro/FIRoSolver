import sys
import random
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel
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

        # Label to display mouse coordinates
        self.coord_label = QLabel("Mouse coordinates: (x, y)", self)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.coord_label)  # Add the label to the layout

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

        # Connect the canvas' motion event to the method that updates the coordinates
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

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

    def on_mouse_move(self, event):
        """Update the label with the current mouse coordinates."""
        if event.inaxes:  # Only if the mouse is over the plot area
            x, y = event.xdata, event.ydata
            self.coord_label.setText(f"Mouse coordinates: ({x:.2f}, {y:.2f})")
        else:
            self.coord_label.setText("Mouse coordinates: (x, y)")  # Reset when outside plot area

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = PlotWindow()
    main.show()
    sys.exit(app.exec())

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from plotter import Plotter  # Ensure correct import

class DraggableLine(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draggable Line Example")

        # Create the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)


    # Create the plot and canvas
        self.plotter = Plotter(self)
        self.canvas = FigureCanvas(self.plotter.fig)
        self.layout.addWidget(self.canvas)
        
        # Create a slider for the Gaussian width
        self.gaussian_width_slider = QSlider(Qt.Orientation.Horizontal, self.main_widget)
        self.gaussian_width_slider.setMinimum(1)
        self.gaussian_width_slider.setMaximum(10)
        self.gaussian_width_slider.setValue(5)
        self.gaussian_width_slider.setSingleStep(1)
        self.gaussian_width_label = QLabel('Gaussian Width', self.main_widget)
        self.layout.addWidget(self.gaussian_width_label)
        self.layout.addWidget(self.gaussian_width_slider)

        # Create a slider for the median filter kernel size
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal, self.main_widget)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setMaximum(21)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setSingleStep(2)
        self.kernel_label = QLabel('Median Filter Kernel Size', self.main_widget)
        self.layout.addWidget(self.kernel_label)
        self.layout.addWidget(self.kernel_slider)

        # Create buttons for smoothen, undo, redo, and selection mode
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.smoothen_button = QPushButton("Smoothen", self.main_widget)
        self.smoothen_button.clicked.connect(self.smoothen_plot)
        self.button_layout.addWidget(self.smoothen_button)

        self.undo_button = QPushButton("Undo", self.main_widget)
        self.undo_button.clicked.connect(self.undo_plot)
        self.button_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo", self.main_widget)
        self.redo_button.clicked.connect(self.redo_plot)
        self.button_layout.addWidget(self.redo_button)

        self.selection_mode_button = QPushButton("Selection Mode", self.main_widget)
        self.selection_mode_button.clicked.connect(self.toggle_selection_mode)
        self.button_layout.addWidget(self.selection_mode_button)

        # Create a slider to set the flat level
        self.flat_level_slider = QSlider(Qt.Orientation.Horizontal, self.main_widget)
        self.flat_level_slider.setMinimum(-60)
        self.flat_level_slider.setMaximum(0)
        self.flat_level_slider.setValue(0)
        self.flat_level_slider.setSingleStep(1)
        self.flat_level_label = QLabel('Flat Level (dB)', self.main_widget)
        self.layout.addWidget(self.flat_level_label)
        self.layout.addWidget(self.flat_level_slider)

        # Apply changes button
        self.apply_flatten_button = QPushButton("Apply Flatten", self.main_widget)
        self.apply_flatten_button.clicked.connect(self.apply_flatten)
        self.layout.addWidget(self.apply_flatten_button)

       

        # Connect event handlers for the plot
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Set up close event handler
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.destroyed.connect(self.on_closing)

    def smoothen_plot(self):
        self.plotter.smoothen_plot()

    def undo_plot(self):
        self.plotter.undo_plot()

    def redo_plot(self):
        self.plotter.redo_plot()

    def toggle_selection_mode(self):
        self.plotter.toggle_selection_mode()

    def apply_flatten(self):
        self.plotter.apply_flatten()

    def on_motion(self, event):
        self.plotter.on_motion(event)

    def on_click(self, event):
        self.plotter.on_click(event)

    def on_release(self, event):
        self.plotter.on_release(event)

    def on_closing(self):
        # Perform additional cleanup
        plt.close(self.plotter.fig)  # Close the Matplotlib figure

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = DraggableLine()
    main_window.show()
    sys.exit(app.exec())

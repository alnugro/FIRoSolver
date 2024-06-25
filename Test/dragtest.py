import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel, QFrame
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import medfilt
from matplotlib.patches import Rectangle

class DraggableLine(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draggable Line Example")

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create control panel
        self.control_frame = QFrame()
        self.control_layout = QVBoxLayout()
        
        # Create sliders
        self.gaussian_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_width_slider.setRange(1, 50)
        self.gaussian_width_slider.setValue(10)
        self.gaussian_width_slider.setSingleStep(1)
        self.gaussian_width_label = QLabel("Gaussian Width")
        self.control_layout.addWidget(self.gaussian_width_label)
        self.control_layout.addWidget(self.gaussian_width_slider)

        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 21)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider_label = QLabel("Median Filter Kernel Size")
        self.control_layout.addWidget(self.kernel_slider_label)
        self.control_layout.addWidget(self.kernel_slider)

        self.flat_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.flat_level_slider.setRange(-100, 100)
        self.flat_level_slider.setValue(0)
        self.flat_level_slider.setSingleStep(1)
        self.flat_level_slider_label = QLabel("Flat Level")
        self.control_layout.addWidget(self.flat_level_slider_label)
        self.control_layout.addWidget(self.flat_level_slider)

        # Create buttons
        self.smoothen_button = QPushButton("Smoothen")
        self.smoothen_button.clicked.connect(self.smoothen_plot)
        self.control_layout.addWidget(self.smoothen_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_plot)
        self.control_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_plot)
        self.control_layout.addWidget(self.redo_button)

        self.selection_mode_button = QPushButton("Selection Mode")
        self.selection_mode_button.clicked.connect(self.toggle_selection_mode)
        self.control_layout.addWidget(self.selection_mode_button)

        self.apply_flatten_button = QPushButton("Apply Flatten")
        self.apply_flatten_button.clicked.connect(self.apply_flatten)
        self.control_layout.addWidget(self.apply_flatten_button)

        self.control_frame.setLayout(self.control_layout)
        self.layout.addWidget(self.control_frame)

        # Create a placeholder for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.draggable_point = None
        self.dragging = False
        self.just_undone = False
        self.history = []
        self.future = []
        self.move_only = False
        self.selection_mode = False
        self.selected_range = None
        self.selection_rect = None

        self.initialize_plot()

        # Connect event handlers
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)

    def initialize_plot(self):
        # Generate initial data
        self.x = np.linspace(0, 10, 100)
        self.y = np.sin(self.x)
        self.save_state()

        # Plot the data
        self.line, = self.ax.plot(self.x, self.y, 'b')
        self.draggable_point, = self.ax.plot([self.x[50]], [self.y[50]], 'ro', picker=5)
        
        self.ax.set_title('Draggable Line Example')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.grid()

        self.canvas.draw()

    def save_state(self):
        # Save the current state for undo functionality
        self.history.append((self.x.copy(), self.y.copy()))
        # Clear the future stack when a new state is saved
        self.future.clear()
        print("State saved. History length:", len(self.history))

    def undo_plot(self):
        if len(self.history) > 1:
            # Move the current state to the future stack
            self.future.append(self.history.pop())
            # Restore the previous state
            self.x, self.y = copy.deepcopy(self.history[-1])
            self.just_undone = True
            print("Undo performed. History length:", len(self.history), "Future length:", len(self.future))
            self.redraw_plot()

    def redo_plot(self):
        if self.future:
            # Restore the next state from the future stack
            self.history.append(self.future.pop())
            self.x, self.y = copy.deepcopy(self.history[-1])
            self.just_undone = False
            print("Redo performed. History length:", len(self.history), "Future length:", len(self.future))
            self.redraw_plot()

    def toggle_selection_mode(self):
        self.selection_mode = not self.selection_mode
        if self.selection_mode:
            self.selection_mode_button.setText("Exit Selection Mode")
            print("Selection mode enabled.")
        else:
            self.selection_mode_button.setText("Selection Mode")
            print("Selection mode disabled.")

    def on_click(self, event):
        if event.inaxes != self.ax:
            print("no figure set when check if mouse is on line")
            return

        if self.selection_mode:
            self.selected_range = [event.xdata, None]
            self.selection_rect = Rectangle((event.xdata, self.ax.get_ylim()[0]), 0, self.ax.get_ylim()[1] - self.ax.get_ylim()[0], color='gray', alpha=0.3)
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw()
            print(f"Selection started at {event.xdata}.")
        else:
            contains, _ = self.draggable_point.contains(event)
            if event.button == 1 and contains:  # Left click
                self.dragging = True
                self.move_only = False
                print("Left click drag started.")
            elif event.button == 3:  # Right click
                self.dragging = True
                self.move_only = True
                self.update_draggable_point(event, move_only=True)
                print("Right click move started.")

    def on_release(self, event):
        if self.selection_mode and self.selected_range:
            self.selected_range[1] = event.xdata
            self.selection_mode = False
            self.selection_mode_button.setText("Selection Mode")
            if self.selection_rect:
                self.selection_rect.set_width(self.selected_range[1] - self.selected_range[0])
                self.canvas.draw()
            print(f"Selection ended at {event.xdata}. Range: {self.selected_range}")
        elif self.dragging:
            if self.just_undone:
                # Clear the future states after the current state
                print("Clearing future states from history after undo.")
                self.history = self.history[:len(self.history)]
                self.just_undone = False  # Reset the flag since a new action is started
            if not self.move_only:
                self.save_state()
                print("Drag ended and state saved.")
            self.dragging = False
            self.move_only = False
    
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.dragging:
            self.update_draggable_point(event, move_only=self.move_only)
        elif self.selection_mode and self.selected_range:
            if self.selection_rect:
                self.selection_rect.set_width(event.xdata - self.selected_range[0])
                self.canvas.draw()

    def update_draggable_point(self, event, move_only=False):
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        
        if move_only:
            idx = np.argmin(np.abs(self.x - x))
            y = self.y[idx]
            self.draggable_point.set_data([self.x[idx]], [y])
        else:
            idx = np.argmin(np.abs(self.x - x))
            self.y[idx] = y
            self.draggable_point.set_data([self.x[idx]], [y])

        self.redraw_plot()

    def smoothen_plot(self):
        kernel_size = self.kernel_slider.value()  # Get the kernel size from the slider
        self.y = medfilt(self.y, kernel_size)
        self.save_state()  # Save state after smoothing
        self.redraw_plot()

    def apply_flatten(self):
        if not self.selected_range:
            print("No range selected.")
            return
        
        start, end = self.selected_range
        if start is None or end is None:
            print("Incomplete range selection.")
            return

        # Ensure start is less than end
        start, end = sorted([start, end])
        
        target_y = self.flat_level_slider.value() / 100.0  # Adjust to match the scale
        indices = (self.x >= start) & (self.x <= end)
        self.y[indices] = target_y
        
        self.save_state()  # Save state after flattening
        self.redraw_plot()
        print(f"Applied flattening from {start} to {end} at level {target_y}.")
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None

    def redraw_plot(self):
        # Clear the previous plot
        self.ax.clear()

        # Plot the updated line
        self.ax.plot(self.x, self.y, 'b')

        # Update the draggable point
        current_x, current_y = self.draggable_point.get_data()
        idx = np.argmin(np.abs(self.x - current_x))
        self.draggable_point.set_data([self.x[idx]], [self.y[idx]])
        self.ax.plot([self.x[idx]], [self.y[idx]], 'ro', picker=5)

        self.ax.set_title('Draggable Line Example')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.grid()

        self.canvas.draw()

    def on_closing(self):
        # Disconnect matplotlib event handlers
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_click)
        self.canvas.mpl_disconnect(self.cid_release)
        # Perform additional cleanup
        plt.close(self.fig)  # Close the Matplotlib figure


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = DraggableLine()
    main_window.show()
    sys.exit(app.exec())

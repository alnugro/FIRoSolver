import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QSlider, QLabel, QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

class DraggableLine(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draggable Lines Example")

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        # Remove or comment out this line to avoid setting multiple layouts
        # self.layout = QVBoxLayout(self.main_widget)

        # Create control panel
        self.control_frame = QFrame()
        self.control_layout = QVBoxLayout()

        # Create Gaussian width slider
        self.gaussian_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.gaussian_width_slider.setRange(1, 50)
        self.gaussian_width_slider.setValue(10)
        self.gaussian_width_slider.setSingleStep(1)
        self.gaussian_width_label = QLabel("Gaussian Width")
        self.control_layout.addWidget(self.gaussian_width_label)
        self.control_layout.addWidget(self.gaussian_width_slider)

        # Create kernel slider
        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 21)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider_label = QLabel("Median Filter Kernel Size")
        self.control_layout.addWidget(self.kernel_slider_label)
        self.control_layout.addWidget(self.kernel_slider)

        # Create flat level slider
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

        # New buttons for delete and mirror functionalities
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected)
        self.control_layout.addWidget(self.delete_button)

        self.mirror_upper_to_lower_button = QPushButton("Mirror Upper to Lower")
        self.mirror_upper_to_lower_button.clicked.connect(self.mirror_upper_to_lower)
        self.control_layout.addWidget(self.mirror_upper_to_lower_button)

        self.mirror_lower_to_upper_button = QPushButton("Mirror Lower to Upper")
        self.mirror_lower_to_upper_button.clicked.connect(self.mirror_lower_to_upper)
        self.control_layout.addWidget(self.mirror_lower_to_upper_button)

        # Day/Night mode toggle button
        self.day_night_button = QPushButton("Switch to Night Mode")
        self.day_night_button.clicked.connect(self.toggle_day_night_mode)
        self.control_layout.addWidget(self.day_night_button)
        self.day_night = False  # Day mode by default

        # Reset plot button
        self.reset_button = QPushButton("Reset Plot")
        self.reset_button.clicked.connect(self.reset_plot)
        self.control_layout.addWidget(self.reset_button)

        # Reset draggable points button
        self.reset_draggable_points_button = QPushButton("Reset Draggable Points")
        self.reset_draggable_points_button.clicked.connect(self.reset_draggable_points)
        self.control_layout.addWidget(self.reset_draggable_points_button)

        self.control_frame.setLayout(self.control_layout)

        # Create a layout for the plot and mouse position label
        self.plot_layout = QVBoxLayout()
        self.plot_widget = QWidget()
        self.plot_widget.setLayout(self.plot_layout)

        # Create a placeholder for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.plot_layout.addWidget(self.canvas)

        # Add the mouse position label
        self.position_label = QLabel("Mouse Position: x = N/A, y = N/A")
        self.plot_layout.addWidget(self.position_label)

        # Add control panel and plot layout to the main layout
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.addWidget(self.control_frame)
        self.main_layout.addWidget(self.plot_widget)
        # No need to call self.main_widget.setLayout(self.main_layout) again

        self.dragging_point = None
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
        self.middle_y = np.sin(self.x)
        self.middle_y[4:5] = np.nan  # Introduce NaN values

        # Define upper and lower bounds
        self.offset = 0.5  # Initial offset value between middle line and upper/lower lines
        self.upper_y = self.middle_y + self.offset
        self.lower_y = self.middle_y - self.offset

        # Store initial data for reset functionality
        self.initial_middle_y = self.middle_y.copy()
        self.initial_upper_y = self.upper_y.copy()
        self.initial_lower_y = self.lower_y.copy()

        self.save_state()

        # Plot the data
        self.middle_line, = self.ax.plot(self.x, self.middle_y, 'b', label='Middle Line')
        self.upper_line, = self.ax.plot(self.x, self.upper_y, 'g', label='Upper Line')
        self.lower_line, = self.ax.plot(self.x, self.lower_y, 'r', label='Lower Line')

        # Create draggable points
        self.middle_draggable_point, = self.ax.plot([self.x[50]], [self.middle_y[50]], 'bo', picker=5)
        self.upper_draggable_point, = self.ax.plot([self.x[50]], [self.upper_y[50]], 'go', picker=5)
        self.lower_draggable_point, = self.ax.plot([self.x[50]], [self.lower_y[50]], 'ro', picker=5)

        self.ax.set_title('Draggable Lines Example')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.grid()
        self.ax.legend()

        self.canvas.draw()

    def reset_draggable_points(self):
        # Find indices where middle_y is not NaN
        valid_indices = np.where(~np.isnan(self.middle_y))[0]
        if len(valid_indices) == 0:
            print("No valid data points to reset draggable points.")
            return

        # Choose an index from the valid indices (e.g., the middle valid index)
        idx = valid_indices[len(valid_indices) // 2]

        # Update the positions of the draggable points
        self.middle_draggable_point.set_data([self.x[idx]], [self.middle_y[idx]])
        self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
        self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])

        # Redraw the plot to reflect the changes
        self.redraw_plot()

        print(f"Draggable points reset to index {idx}.")


    def save_state(self):
        # Save the current state for undo functionality
        self.history.append((self.x.copy(), self.middle_y.copy(), self.upper_y.copy(), self.lower_y.copy()))
        # Clear the future stack when a new state is saved
        self.future.clear()
        print("State saved. History length:", len(self.history))

    def toggle_day_night_mode(self):
        self.day_night = not self.day_night
        if self.day_night:
            plt.style.use('dark_background')
            self.day_night_button.setText("Switch to Day Mode")
            print("Switched to Night Mode.")
        else:
            plt.style.use('default')
            self.day_night_button.setText("Switch to Night Mode")
            print("Switched to Day Mode.")
        self.recreate_plot()

    def recreate_plot(self):
        # Disconnect events
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_click)
        self.canvas.mpl_disconnect(self.cid_release)
        # Close the figure
        plt.close(self.fig)
        # Create a new figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas.figure = self.fig
        self.canvas.draw_idle()
        # Reconnect events
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        # Redraw the plot
        self.redraw_plot()

    def reset_plot(self):
        self.middle_y = self.initial_middle_y.copy()
        self.upper_y = self.initial_upper_y.copy()
        self.lower_y = self.initial_lower_y.copy()
        self.history = []
        self.future = []
        self.save_state()
        self.redraw_plot()
        print("Plot reset to initial state.")

    def undo_plot(self):
        if len(self.history) > 1:
            # Move the current state to the future stack
            self.future.append(self.history.pop())
            # Restore the previous state
            self.x, self.middle_y, self.upper_y, self.lower_y = copy.deepcopy(self.history[-1])
            self.just_undone = True
            print("Undo performed. History length:", len(self.history), "Future length:", len(self.future))
            self.redraw_plot()

    def redo_plot(self):
        if self.future:
            # Restore the next state from the future stack
            self.history.append(self.future.pop())
            self.x, self.middle_y, self.upper_y, self.lower_y = copy.deepcopy(self.history[-1])
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
            print("No figure set when check if mouse is on line")
            return

        if self.selection_mode:
            self.selected_range = [event.xdata, None]
            self.selection_rect = Rectangle((event.xdata, self.ax.get_ylim()[0]), 0, self.ax.get_ylim()[1] - self.ax.get_ylim()[0], color='gray', alpha=0.3)
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw()
            print(f"Selection started at {event.xdata}.")
        else:
            # Check which draggable point is being clicked
            contains_middle, _ = self.middle_draggable_point.contains(event)
            contains_upper, _ = self.upper_draggable_point.contains(event)
            contains_lower, _ = self.lower_draggable_point.contains(event)

            if event.button == 1:
                if contains_middle:
                    self.dragging = True
                    self.dragging_point = 'middle'
                    self.move_only = False
                    print("Left click drag started on middle line.")
                elif contains_upper:
                    self.dragging = True
                    self.dragging_point = 'upper'
                    self.move_only = False
                    print("Left click drag started on upper line.")
                elif contains_lower:
                    self.dragging = True
                    self.dragging_point = 'lower'
                    self.move_only = False
                    print("Left click drag started on lower line.")
            elif event.button == 3:
                # Determine which draggable point is closest to the click
                distances = []
                for point, name in [(self.middle_draggable_point, 'middle'),
                                    (self.upper_draggable_point, 'upper'),
                                    (self.lower_draggable_point, 'lower')]:
                    xdata, ydata = point.get_data()
                    distance = np.hypot(event.xdata - xdata[0], event.ydata - ydata[0])
                    distances.append((distance, name))
                distances.sort()
                closest_point = distances[0][1]
                self.dragging = True
                self.dragging_point = closest_point
                self.move_only = True
                self.update_draggable_point(event, move_only=True)
                print(f"Right click move started on {closest_point} line.")

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
            self.dragging_point = None  # Reset the dragging point

    def on_motion(self, event):
        if event.inaxes != self.ax:
            self.position_label.setText("Mouse Position: x = N/A, y = N/A")
            return

        if event.xdata is not None and event.ydata is not None:
            x = event.xdata
            y = event.ydata
            self.position_label.setText(f"Mouse Position: x = {x:.2f}, y = {y:.2f}")
        else:
            self.position_label.setText("Mouse Position: x = N/A, y = N/A")

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
        if np.isnan(x) or np.isnan(y):
            return

        idx = np.argmin(np.abs(self.x - x))
        if move_only:
            # Move the draggable point horizontally without changing the data
            if self.dragging_point == 'middle':
                y_val = self.middle_y[idx]
                self.middle_draggable_point.set_data([self.x[idx]], [y_val])
            elif self.dragging_point == 'upper':
                y_val = self.upper_y[idx]
                self.upper_draggable_point.set_data([self.x[idx]], [y_val])
            elif self.dragging_point == 'lower':
                y_val = self.lower_y[idx]
                self.lower_draggable_point.set_data([self.x[idx]], [y_val])
        else:
            # Apply Gaussian smoothing to the change
            gaussian_width = self.gaussian_width_slider.value()
            if gaussian_width < 1:
                gaussian_width = 1

            sigma = gaussian_width / 2.0  # Adjust sigma based on slider
            window_size = int(gaussian_width * 3)  # For Gaussian, values beyond 3*sigma are negligible
            start_idx = max(0, idx - window_size)
            end_idx = min(len(self.x), idx + window_size + 1)
            indices = np.arange(start_idx, end_idx)

            # Calculate Gaussian weights
            distances = indices - idx
            weights = np.exp(-0.5 * (distances / sigma) ** 2)
            weights /= weights.sum()  # Normalize weights

            # Create masks for NaN values
            middle_nan_mask = np.isnan(self.middle_y[indices])
            upper_nan_mask = np.isnan(self.upper_y[indices])
            lower_nan_mask = np.isnan(self.lower_y[indices])

            if self.dragging_point == 'middle':
                delta_y = y - self.middle_y[idx]
                # Apply weighted delta_y to middle_y where not NaN
                self.middle_y[indices] = np.where(
                    middle_nan_mask,
                    self.middle_y[indices],
                    self.middle_y[indices] + delta_y * weights
                )
                # Move upper and lower lines accordingly where not NaN
                self.upper_y[indices] = np.where(
                    upper_nan_mask,
                    self.upper_y[indices],
                    self.upper_y[indices] + delta_y * weights
                )
                self.lower_y[indices] = np.where(
                    lower_nan_mask,
                    self.lower_y[indices],
                    self.lower_y[indices] + delta_y * weights
                )
            elif self.dragging_point == 'upper':
                delta_y = y - self.upper_y[idx]
                # Apply weighted delta_y to upper_y where not NaN
                new_upper_y = np.where(
                    upper_nan_mask,
                    self.upper_y[indices],
                    self.upper_y[indices] + delta_y * weights
                )
                # Ensure upper_y >= middle_y
                self.upper_y[indices] = np.where(
                    upper_nan_mask,
                    self.upper_y[indices],
                    np.maximum(new_upper_y, self.middle_y[indices])
                )
            elif self.dragging_point == 'lower':
                delta_y = y - self.lower_y[idx]
                # Apply weighted delta_y to lower_y where not NaN
                new_lower_y = np.where(
                    lower_nan_mask,
                    self.lower_y[indices],
                    self.lower_y[indices] + delta_y * weights
                )
                # Ensure lower_y <= middle_y
                self.lower_y[indices] = np.where(
                    lower_nan_mask,
                    self.lower_y[indices],
                    np.minimum(new_lower_y, self.middle_y[indices])
                )
            else:
                return  # No valid dragging point

            # Update draggable point positions
            if self.dragging_point == 'middle':
                self.middle_draggable_point.set_data([self.x[idx]], [self.middle_y[idx]])
                self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
                self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])
            elif self.dragging_point == 'upper':
                self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
            elif self.dragging_point == 'lower':
                self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])

        self.redraw_plot()

    def smoothen_plot(self):
        kernel_size = self.kernel_slider.value()  # Get the kernel size from the slider
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd

        if self.selected_range:
            # Smooth only the selected range
            start, end = sorted(self.selected_range)
            indices = (self.x >= start) & (self.x <= end)
            # Smooth the selected range
            self.middle_y = self.nanmedfilt_partial(self.middle_y, kernel_size, indices)
            self.upper_y = self.nanmedfilt_partial(self.upper_y, kernel_size, indices)
            self.lower_y = self.nanmedfilt_partial(self.lower_y, kernel_size, indices)
            print(f"Applied smoothing to selected range: {start} to {end}")
            # Clear selection
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None
                self.selected_range = None
                self.canvas.draw()
        else:
            # Smooth the entire data
            self.middle_y = self.nanmedfilt(self.middle_y, kernel_size)
            self.upper_y = self.nanmedfilt(self.upper_y, kernel_size)
            self.lower_y = self.nanmedfilt(self.lower_y, kernel_size)
            print("Applied smoothing to entire data")

        self.save_state()  # Save state after smoothing
        self.redraw_plot()

    def nanmedfilt(self, y, kernel_size):
        # Custom median filter that ignores NaN values and preserves NaNs
        half_window = kernel_size // 2
        y_padded = np.pad(y, (half_window, half_window), mode='reflect')
        y_smooth = np.empty_like(y)

        for i in range(len(y)):
            if np.isnan(y[i]):
                y_smooth[i] = np.nan  # Preserve NaN values
                continue
            start_idx = i
            end_idx = i + kernel_size
            window = y_padded[start_idx:end_idx]
            # Ignore NaN values in the window
            window_valid = window[~np.isnan(window)]
            if len(window_valid) > 0:
                y_smooth[i] = np.median(window_valid)
            else:
                y_smooth[i] = np.nan  # If all values are NaN, result is NaN

        return y_smooth

    def nanmedfilt_partial(self, y, kernel_size, indices):
        # Apply median filter only on specified indices
        y_smooth = y.copy()
        idx_list = np.where(indices)[0]
        half_window = kernel_size // 2
        y_padded = np.pad(y, (half_window, half_window), mode='reflect')

        for idx in idx_list:
            if np.isnan(y[idx]):
                continue  # Preserve NaN values
            start_idx = idx
            end_idx = idx + kernel_size
            window = y_padded[start_idx:end_idx]
            # Ignore NaN values in the window
            window_valid = window[~np.isnan(window)]
            if len(window_valid) > 0:
                y_smooth[idx] = np.median(window_valid)
            else:
                y_smooth[idx] = np.nan  # If all values are NaN, result is NaN

        return y_smooth

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

        # Create masks for NaN values
        middle_nan_mask = np.isnan(self.middle_y[indices])
        upper_nan_mask = np.isnan(self.upper_y[indices])
        lower_nan_mask = np.isnan(self.lower_y[indices])

        delta_y = target_y - self.middle_y[indices]
        # Apply flattening where not NaN
        self.middle_y[indices] = np.where(
            middle_nan_mask,
            self.middle_y[indices],
            target_y
        )
        self.upper_y[indices] = np.where(
            upper_nan_mask,
            self.upper_y[indices],
            self.upper_y[indices] + delta_y
        )
        self.lower_y[indices] = np.where(
            lower_nan_mask,
            self.lower_y[indices],
            self.lower_y[indices] + delta_y
        )

        self.save_state()  # Save state after flattening
        self.redraw_plot()
        print(f"Applied flattening from {start} to {end} at level {target_y}.")
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.selected_range = None
            self.canvas.draw()

    def delete_selected(self):
        if not self.selected_range:
            print("No range selected to delete.")
            return

        start, end = self.selected_range
        if start is None or end is None:
            print("Incomplete range selection.")
            return

        start, end = sorted([start, end])
        indices = (self.x >= start) & (self.x <= end)

        # Set selected data to NaN
        self.middle_y[indices] = np.nan
        self.upper_y[indices] = np.nan
        self.lower_y[indices] = np.nan

        self.save_state()
        self.redraw_plot()
        print(f"Deleted data from {start} to {end}.")
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.selected_range = None
            self.canvas.draw()

    def mirror_upper_to_lower(self):
        # Calculate the difference between upper and middle lines
        delta = self.upper_y - self.middle_y
        # Mirror to lower line
        self.lower_y = self.middle_y - delta

        self.save_state()
        self.redraw_plot()
        print("Mirrored upper bound to lower bound.")

    def mirror_lower_to_upper(self):
        # Calculate the difference between middle and lower lines
        delta = self.middle_y - self.lower_y
        # Mirror to upper line
        self.upper_y = self.middle_y + delta

        self.save_state()
        self.redraw_plot()
        print("Mirrored lower bound to upper bound.")

    def redraw_plot(self):
        # Clear the previous plot
        self.ax.clear()

        # Plot the updated lines
        self.ax.plot(self.x, self.middle_y, 'b', label='Middle Line')
        self.ax.plot(self.x, self.upper_y, 'g', label='Upper Line')
        self.ax.plot(self.x, self.lower_y, 'r', label='Lower Line')

        # Update the draggable points
        # For middle line
        middle_x, _ = self.middle_draggable_point.get_data()
        middle_idx = np.argmin(np.abs(self.x - middle_x))
        self.middle_draggable_point.set_data([self.x[middle_idx]], [self.middle_y[middle_idx]])
        self.ax.plot([self.x[middle_idx]], [self.middle_y[middle_idx]], 'bo', picker=5)
        # For upper line
        upper_x, _ = self.upper_draggable_point.get_data()
        upper_idx = np.argmin(np.abs(self.x - upper_x))
        self.upper_draggable_point.set_data([self.x[upper_idx]], [self.upper_y[upper_idx]])
        self.ax.plot([self.x[upper_idx]], [self.upper_y[upper_idx]], 'go', picker=5)
        # For lower line
        lower_x, _ = self.lower_draggable_point.get_data()
        lower_idx = np.argmin(np.abs(self.x - lower_x))
        self.lower_draggable_point.set_data([self.x[lower_idx]], [self.lower_y[lower_idx]])
        self.ax.plot([self.x[lower_idx]], [self.lower_y[lower_idx]], 'ro', picker=5)

        self.ax.set_title('Draggable Lines Example')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.grid()
        self.ax.legend()

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

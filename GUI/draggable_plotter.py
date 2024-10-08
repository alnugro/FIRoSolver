import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import medfilt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle

from matplotlib.figure import Figure
from matplotlib.axes import Axes

class DraggablePlotter:
    def __init__(self, fig : Figure, ax : Axes, canvas : FigureCanvasQTAgg, day_night, position_label, app = None):
        # Create a placeholder for the plot
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.app = app

        self.sampling_rate = app.sampling_rate_box.value()
        self.selection_mode = app.selection_mode
        self.gaussian_width = app.gausian_slid.value()
        self.smoothing_kernel = app.smoothing_kernel_slid.value()
        self.flat_level = app.flatten_box.value()
        self.ignore_lowerbound_point = 10 ** (float(app.ignore_lowerbound_box.value())/20)

        self.dragging_point = None
        self.dragging = False
        self.just_undone = False
        self.history = []
        self.future = []
        self.move_only = False
        
        self.selected_range = None
        self.selection_rect = None

        self.sampling_rate_used = None
        self.is_nyquist = True

        # Connect event handlers
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)

        self.day_night = day_night

        self.position_label = position_label

    def initialize_plot(self, xdata, middle_y, upper_y, lower_y):

        self.x = np.copy(xdata)
        self.middle_y = np.copy(middle_y)
        self.upper_y = np.copy(upper_y)
        self.lower_y = np.copy(lower_y)


        self.save_state()
        # Plot the data
        self.middle_line, = self.ax.plot(self.x, self.middle_y, 'b', label='Magnitude')
        self.upper_line, = self.ax.plot(self.x, self.upper_y, 'g', label='Upperbound')
        self.lower_line, = self.ax.plot(self.x, self.lower_y, 'r', label='Lowerbound')
        
        self.ignore_line, = self.ax.plot([0,1], [self.ignore_lowerbound_point,self.ignore_lowerbound_point], 'y',linestyle='--', label='Ignore Lowerb.')

        not_nan_index = np.where(~np.isnan(self.middle_y))[0][0]

        # Create draggable points
        self.middle_draggable_point, = self.ax.plot([self.x[not_nan_index]], [self.middle_y[not_nan_index]], 'bo', picker=5)
        self.upper_draggable_point, = self.ax.plot([self.x[not_nan_index]], [self.upper_y[not_nan_index]], 'go', picker=5)
        self.lower_draggable_point, = self.ax.plot([self.x[not_nan_index]], [self.lower_y[not_nan_index]], 'ro', picker=5)

        self.ax.legend()

        self.canvas.draw()


    def update_position_label(self):
        x_val = self.x[self.current_idx]
        if self.dragging_point == 'middle':
            y_val = self.middle_y[self.current_idx]
        elif self.dragging_point == 'upper':
            y_val = self.upper_y[self.current_idx]
        elif self.dragging_point == 'lower':
            y_val = self.lower_y[self.current_idx]
        else:
            # Default to middle line if no point is being dragged
            y_val = self.middle_y[self.current_idx]
            self.dragging_point = 'middle'
        if np.isnan(y_val):
            self.position_label.setText("Point Pos.: x = N/A, y = N/A")
        else:
            self.position_label.setText(f"Point Pos.: x = {x_val:.6f}, y = {y_val:.6f}")

    def save_state(self):
        # Save the current state for undo functionality
        self.history.append((self.x.copy(), self.middle_y.copy(), self.upper_y.copy(), self.lower_y.copy()))
        # Clear the future stack when a new state is saved
        self.future.clear()
        print("State saved. History length:", len(self.history))

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
        

        idx = np.argmin(np.abs(self.x - x))
        if np.isnan(self.upper_y[idx]) or np.isnan(self.lower_y[idx]) or np.isnan(self.middle_y[idx]):
            return
        self.current_idx = idx  # Update the current index of the draggable point

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
            # Update the draggable point position label
            self.update_position_label()
        else:
            # Apply Gaussian smoothing to the change
            gaussian_width = self.app.gausian_slid.value()
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
                # Update the draggable point positions
                self.middle_draggable_point.set_data([self.x[idx]], [self.middle_y[idx]])
                self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
                self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])
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
                # Update the draggable point position
                self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
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
                # Update the draggable point position
                self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])
            else:
                return  # No valid dragging point

            # Update the draggable point position label
            self.update_position_label()

        self.redraw_plot()

    def smoothen_plot(self):
        print(self.app.smoothing_kernel_slid.value() )
        kernel_size = self.app.smoothing_kernel_slid.value()  # Get the kernel size from the slider
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

        target_y = self.app.flatten_box.value()# Adjust to match the scale
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

    def redraw_plot(self):
        lower_current_xlim, upper_current_xlim = self.ax.get_xlim()
        lower_current_ylim, upper_current_ylim = self.ax.get_ylim()

        # Clear the previous plot
        self.ax.clear()

        # Plot the updated lines
        self.ax.plot(self.x, self.middle_y, 'b', label='Magnitude')
        self.ax.plot(self.x, self.upper_y, 'g', label='Upperbound')
        self.ax.plot(self.x, self.lower_y, 'r', label='Lowerbound')

        self.ax.plot([0,self.x[-1]], [self.ignore_lowerbound_point,self.ignore_lowerbound_point], 'y',linestyle='dotted', label='Ignore Lowerb.')
        
        self.ax.set_xlim([lower_current_xlim, upper_current_xlim])
        self.ax.set_ylim([lower_current_ylim, upper_current_ylim])

        if self.day_night:
            pass
        else:
            self.ax.grid(True)

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

        # Update the draggable point position label
        self.update_position_label()

        self.ax.legend()

        self.canvas.draw()


    def disconnect(self):
        # Disconnect matplotlib event handlers
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_click)
        self.canvas.mpl_disconnect(self.cid_release)
    
    def update_plotter_data(self, data_dict, redraw = True):
        for data_key, data_value in data_dict.items():
            if hasattr(self, data_key):
                setattr(self, data_key, data_value)
        if redraw:
            self.redraw_plot()

    def nyquist_to_sampling(self):
        if not(self.is_nyquist):
            return 1
        self.x = np.array([xdat * self.sampling_rate if not np.isnan(xdat) else np.nan for xdat in self.x])
        self.sampling_rate_used = self.sampling_rate
        self.is_nyquist = False
        self.save_state()
        self.redraw_plot()

        return 0

    def sampling_to_nyquist(self):
        if self.is_nyquist:
            return 1
        self.x = np.array([xdat / self.sampling_rate_used if not np.isnan(xdat) else np.nan for xdat in self.x])
        self.is_nyquist = True

        self.save_state()
        self.redraw_plot()

        return 0
    
    def reset_draggable_points(self):
        # Find indices where middle_y is not NaN
        valid_indices = np.where(~np.isnan(self.middle_y))[0]
        if len(valid_indices) == 0:
            print("No valid data points to reset draggable points.")
            return 1

        # Choose an index from the valid indices (e.g., the middle valid index)
        idx = valid_indices[len(valid_indices) // 2]

        # Update the positions of the draggable points
        self.middle_draggable_point.set_data([self.x[idx]], [self.middle_y[idx]])
        self.upper_draggable_point.set_data([self.x[idx]], [self.upper_y[idx]])
        self.lower_draggable_point.set_data([self.x[idx]], [self.lower_y[idx]])

        # Redraw the plot to reflect the changes
        self.redraw_plot()
        print(f"Draggable points reset to index {idx}.")

        return 0
    
    def get_plot_data(self):
        return self.x, self.middle_y, self.lower_y, self.upper_y
    
    def get_history(self):
        return self.history
    
    def set_history(self, history):
        self.history = history

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from matplotlib.patches import Rectangle

class Plotter:
    def __init__(self, app=None):
        self.app = app
        self.fig, self.ax = plt.subplots(figsize=(8, 4))

        self.x = np.arange(0.0, 1.0, 0.01)
        self.y = np.sin(self.x)

        if app:
            self.history = [(self.x.copy(), self.y.copy())]
            self.future = []

        self.line, = self.ax.plot(self.x, self.y, 'b')
        self.draggable_point, = self.ax.plot([self.x[0]], [self.y[0]], 'ro', picker=5)

        self.ax.set_title('Draggable Line Example')
        self.ax.set_xlabel('Normalized Frequency (0-1)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-60, 5])
        self.ax.grid()

        # Initialize additional attributes
        self.dragging = False
        self.move_only = False
        self.selection_mode = False
        self.selected_range = None
        self.selection_rect = None
        self.just_undone = False

    def redraw_plot(self):
        self.ax.clear()
        self.ax.plot(self.x, self.y, 'b')
        current_x, current_y = self.draggable_point.get_data()
        idx = np.argmin(np.abs(self.x - current_x))
        self.draggable_point.set_data([self.x[idx]], [self.y[idx]])
        self.ax.plot([self.x[idx]], [self.y[idx]], 'ro', picker=5)
        self.ax.set_title('Draggable Line Example')
        self.ax.set_xlabel('Normalized Frequency (0-1)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-60, 5])
        self.ax.grid()
        if self.app:
            self.app.canvas.draw()

    def smoothen_plot(self):
        kernel_size = self.app.kernel_slider.value()
        self.y = medfilt(self.y, kernel_size)
        self.save_state()
        self.redraw_plot()

    def undo_plot(self):
        if len(self.history) > 1:
            self.future.append(self.history.pop())
            self.x, self.y = copy.deepcopy(self.history[-1])
            self.just_undone = True
            self.redraw_plot()

    def redo_plot(self):
        if self.future:
            self.history.append(self.future.pop())
            self.x, self.y = copy.deepcopy(self.history[-1])
            self.just_undone = False
            self.redraw_plot()

    def toggle_selection_mode(self):
        self.selection_mode = not self.selection_mode
        if self.selection_mode:
            self.app.selection_mode_button.setText("Exit Selection Mode")
        else:
            self.app.selection_mode_button.setText("Selection Mode")

    def apply_flatten(self):
        if not self.selected_range:
            return
        start, end = self.selected_range
        start, end = sorted([start, end])
        target_y = self.app.flat_level_slider.value()
        indices = (self.x >= start) & (self.x <= end)
        self.y[indices] = target_y
        self.save_state()
        self.redraw_plot()
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None

    def save_state(self):
        self.history.append((self.x.copy(), self.y.copy()))
        self.future.clear()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.selection_mode:
            self.selected_range = [event.xdata, None]
            self.selection_rect = Rectangle((event.xdata, self.ax.get_ylim()[0]), 0, self.ax.get_ylim()[1] - self.ax.get_ylim()[0], color='gray', alpha=0.3)
            self.ax.add_patch(self.selection_rect)
            self.app.canvas.draw()
        else:
            contains, _ = self.draggable_point.contains(event)
            if event.button == 1 and contains:
                self.dragging = True
                self.move_only = False
            elif event.button == 3 and contains:
                self.dragging = True
                self.move_only = True
                self.update_draggable_point(event, move_only=True)

    def on_release(self, event):
        if self.selection_mode and self.selected_range:
            self.selected_range[1] = event.xdata
            self.selection_mode = False
            self.app.selection_mode_button.setText("Selection Mode")
            if self.selection_rect:
                self.selection_rect.set_width(self.selected_range[1] - self.selected_range[0])
                self.app.canvas.draw()
        elif self.dragging:
            if self.just_undone:
                self.history = self.history[:len(self.history)]
                self.just_undone = False
            if not self.move_only:
                self.save_state()
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
                self.app.canvas.draw()

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
            x = self.x[idx]
            self.draggable_point.set_data([x], [y])
            sigma = self.app.gaussian_width_slider.value()
            influence = np.exp(-0.5 * ((self.x - x) / sigma) ** 2)
            delta_y = y - self.y[idx]
            self.y += influence * delta_y
        self.redraw_plot()

if __name__ == "__main__":
    plotter = Plotter()
    plt.show()

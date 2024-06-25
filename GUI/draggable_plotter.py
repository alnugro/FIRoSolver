import numpy as np
import matplotlib.pyplot as plt

class DraggablePlotter:
    lock = None  # Only one can be animated at a time

    def __init__(self, line, plotter, color,midpoint):
        self.line = line
        self.plotter = plotter
        self.press = None
        self.background = None
        self.dragging = False
        self.just_undone = False
        self.history = []
        self.future = []
        self.move_only = False
        self.color=color
        self.dot_color='white'
        self.midpoint=midpoint

        if self.color=="gainsboro":
            self.dot_color='white'
        if self.color=="salmon":
            self.dot_color='lightsalmon'    
        if self.color=="aqua":
            self.dot_color='paleturquoise'

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        

        # Initialize draggable point with separate color and marker style
        self.draggable_point, = line.axes.plot([xdata[0]], [ydata[0]], color=self.dot_color, marker='o', picker=5)
        self.line_handle = None
        self.point_handle = None

    def connect(self):
        self.cidpress = self.draggable_point.figure.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cidrelease = self.draggable_point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.draggable_point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_click(self, event):
        if event.inaxes != self.line.axes:
            return
        if DraggablePlotter.lock is not None:
            return
        contains, attrd = self.draggable_point.contains(event)

        if not contains:  # Event not on this instance's draggable point
            return

        self.press = (self.line.get_xdata(), self.line.get_ydata(), event.xdata, event.ydata)
        DraggablePlotter.lock = self

        if event.button == 1 and contains:  # Left click
            self.dragging = True
            self.move_only = False
            print("Left click drag started.")
        elif event.button == 3:  # Right click
            self.dragging = True
            self.move_only = True
            self.update_draggable_point(event, move_only=True)
            print("Right click move started.")

    def on_motion(self, event):
        if DraggablePlotter.lock is not self:
            return

        if event.inaxes != self.line.axes:
            return

        if self.dragging:
            self.update_draggable_point(event, move_only=self.move_only)

    def on_release(self, event):
        if DraggablePlotter.lock is not self:
            return

        if self.dragging:
            self.dragging = False
            self.move_only = False
            self.press = None
            DraggablePlotter.lock = None

    def update_draggable_point(self, event, move_only=False):
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        xdata = self.line.get_xdata()
        ydata = self.line.get_ydata()

        if move_only:
            idx = np.argmin(np.abs(xdata - x))
            y = ydata[idx]
            self.draggable_point.set_data([xdata[idx]], [y])
        else:
            idx = np.argmin(np.abs(xdata - x))
            ydata[idx] = y
            self.line.set_ydata(ydata)
            self.draggable_point.set_data([xdata[idx]], [y])

        self.redraw_plot()

    def redraw_plot(self):
        # Remove the previous line plot if it exists
        if hasattr(self, 'line_handle') and self.line_handle:
            self.line_handle.remove()
        if hasattr(self, 'point_handle') and self.point_handle:
            self.point_handle.remove()

        # Plot the updated line
        self.line_handle, = self.line.axes.plot(self.line.get_xdata(), self.line.get_ydata(), color=self.color)
        current_x, current_y = self.draggable_point.get_data()

        # Update the draggable point
        self.point_handle, = self.line.axes.plot(current_x, current_y, color=self.dot_color, marker='o', picker=5)
        self.line.axes.figure.canvas.draw()

    def get_xdata(self):
        return self.line.get_xdata()
    
    def get_ydata(self):
        return self.line.get_ydata()

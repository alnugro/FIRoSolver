import copy
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import medfilt
from matplotlib.patches import Rectangle

class DraggableLine(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draggable Line Example")

        # Create a frame for the slider and plot
        self.frame = ttk.Frame(self)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a slider for the Gaussian width
        self.gaussian_width_slider = tk.Scale(self.frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label='Gaussian Width')
        self.gaussian_width_slider.set(1.0)
        self.gaussian_width_slider.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Create a slider for the median filter kernel size
        self.kernel_slider = tk.Scale(self.frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL, label='Median Filter Kernel Size')
        self.kernel_slider.set(5)
        self.kernel_slider.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Create buttons for smoothen, undo, redo, and selection mode
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.smoothen_button = ttk.Button(self.button_frame, text="Smoothen", command=self.smoothen_plot)
        self.smoothen_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.undo_button = ttk.Button(self.button_frame, text="Undo", command=self.undo_plot)
        self.undo_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.redo_button = ttk.Button(self.button_frame, text="Redo", command=self.redo_plot)
        self.redo_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.selection_mode_button = ttk.Button(self.button_frame, text="Selection Mode", command=self.toggle_selection_mode)
        self.selection_mode_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create a slider to set the flat level
        self.flat_level_slider = tk.Scale(self.frame, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label='Flat Level')
        self.flat_level_slider.set(0.0)
        self.flat_level_slider.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Apply changes button
        self.apply_flatten_button = ttk.Button(self.frame, text="Apply Flatten", command=self.apply_flatten)
        self.apply_flatten_button.pack(side=tk.TOP, padx=5, pady=5)

        # Create a placeholder for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

        # Set up close event handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        # Connect event handlers
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
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
            self.selection_mode_button.config(text="Exit Selection Mode")
            print("Selection mode enabled.")
        else:
            self.selection_mode_button.config(text="Selection Mode")
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
            self.selection_mode_button.config(text="Selection Mode")
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
            self.draggable_point.set_data([x], [y])
            idx = np.argmin(np.abs(self.x - x))
            sigma = self.gaussian_width_slider.get()
            influence = np.exp(-0.5 * ((self.x - x) / sigma) ** 2)
            delta_y = y - self.y[idx]
            self.y += influence * delta_y

        self.redraw_plot()

    def smoothen_plot(self):
        kernel_size = self.kernel_slider.get()  # Get the kernel size from the slider
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
        
        target_y = self.flat_level_slider.get()
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
        self.fig.canvas.mpl_disconnect(self.cid_motion)
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_release)
        # Perform additional cleanup
        self.quit()  # Stop the Tkinter main loop
        self.destroy()  # Destroy the Tkinter window
        plt.close(self.fig)  # Close the Matplotlib figure

if __name__ == "__main__":
    app = DraggableLine()
    app.mainloop()

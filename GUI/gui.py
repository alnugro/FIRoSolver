import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from plotter import Plotter  # Ensure correct import

class DraggableLine(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draggable Line Example")

        # Create a frame for the slider and plot
        self.frame = ttk.Frame(self)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a slider for the Gaussian width
        self.gaussian_width_slider = tk.Scale(self.frame, from_=0.01, to=0.1, resolution=0.01, orient=tk.HORIZONTAL, label='Gaussian Width')
        self.gaussian_width_slider.set(0.05)
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
        self.flat_level_slider = tk.Scale(self.frame, from_=-60.0, to=0.0, resolution=1.0, orient=tk.HORIZONTAL, label='Flat Level (dB)')
        self.flat_level_slider.set(0.0)
        self.flat_level_slider.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Apply changes button
        self.apply_flatten_button = ttk.Button(self.frame, text="Apply Flatten", command=self.apply_flatten)
        self.apply_flatten_button.pack(side=tk.TOP, padx=5, pady=5)

        # Create the plot and canvas
        self.plotter = Plotter(self)
        self.canvas = FigureCanvasTkAgg(self.plotter.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect event handlers for the plot
        self.cid_motion = self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_click = self.plotter.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.plotter.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Set up close event handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        # Disconnect matplotlib event handlers
        self.plotter.fig.canvas.mpl_disconnect(self.cid_motion)
        self.plotter.fig.canvas.mpl_disconnect(self.cid_click)
        self.plotter.fig.canvas.mpl_disconnect(self.cid_release)
        # Perform additional cleanup
        self.quit()  # Stop the Tkinter main loop
        self.destroy()  # Destroy the Tkinter window
        plt.close(self.plotter.fig)  # Close the Matplotlib figure

if __name__ == "__main__":
    app = DraggableLine()
    app.mainloop()

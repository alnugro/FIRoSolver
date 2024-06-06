import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import freqz, firwin

class DraggableLine(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plot Example")

        # Create a frame for the plot
        self.frame = ttk.Frame(self)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 

        # Create a placeholder for the plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)   
        self.initialize_plot()

        # Set up close event handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_plot(self):
        # Generate initial data (impulse response of a low-pass filter)
        self.numtaps = 101
        self.cutoff = 0.25  # Fixed cutoff frequency
        self.h = firwin(self.numtaps, self.cutoff)
        self.w, self.h_response = freqz(self.h, worN=8000)
        self.x = self.w / np.pi
        self.y = 20 * np.log10(np.abs(self.h_response))
        self.y = np.clip(self.y, -60, None)  # Clip to -60 dB for better visualization

        # Plot the data
        self.line, = self.ax.plot(self.x, self.y, 'b')
        self.draggable_point, = self.ax.plot([self.x[4000]], [self.y[4000]], 'ro', picker=5)
        
        self.ax.set_title('Draggable Line Example')
        self.ax.set_xlabel('Normalized Frequency (0-1)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-60, 5])
        self.ax.grid()

        # Connect event handlers
        self.canvas.draw()

    def on_closing(self):
        # Perform additional cleanup
        plt.close(self.fig)  # Close the Matplotlib figure
        self.destroy()  # Destroy the Tkinter window

if __name__ == "__main__":
    app = DraggableLine()
    app.mainloop()

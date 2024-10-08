import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_plot():
    # Create a figure and a set of subplots
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Plot some example data
    t = [0, 1, 2, 3, 4, 5]
    s = [0, 1, 4, 9, 16, 25]
    ax.plot(t, s)

    ax.set_title("Sample Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")

    return fig

def main():
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Matplotlib in Tkinter")

    # Create a frame for the plot
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Create the plot and embed it in the frame
    fig = create_plot()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()

    # Add the canvas to the Tkinter window
    canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()

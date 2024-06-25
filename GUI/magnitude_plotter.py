import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from matplotlib.patches import Rectangle
from .draggable_plotter import DraggablePlotter

class MagnitudePlotter:
    def __init__(self, app=None):
        plt.style.use('dark_background')
        self.app = app
        self.fig, self.ax = plt.subplots(figsize=(30, 4))
        self.fig.subplots_adjust(left=0.03, bottom=0.1, right=0.98, top=0.98)

        # Array to put the object of Draggable plotter
        self.draggable_lines_mag = []
        self.draggable_lines_upper = []
        self.draggable_lines_lower = []

        # Stuff to run get data
        self.step = None
        self.upper_xdata = np.array([])
        self.upper_ydata = np.array([])
        self.lower_ydata = np.array([])


        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([-60, 5])
        self.ax.grid()

        # Initialize additional attributes
        self.selection_mode = False
        self.selected_range = None
        self.selection_rect = None

    def initiate_plot(self, table_data):
        highest_upper=0
        lowest_lower=0

        table_data = table_data[table_data[:, 3].argsort()]  # Sort table_data by the fourth column (start frequency)

        for row in table_data:
            magnitude, lower_bound, upper_bound, start_freq, end_freq = row

            #highest x and y
            

            if highest_upper < magnitude+upper_bound:
                print("highest upper called")
                highest_upper = magnitude+upper_bound + 5

            if lowest_lower > magnitude-lower_bound:
                print("lowest lower called")
                lowest_lower=magnitude-lower_bound - 10
            
            self.ax.set_ylim([lowest_lower, highest_upper])



            # Generate points between start and end frequencies with 10^-8 distance between each points
            freqs = np.arange(start_freq, end_freq, 0.001)
            midpoint=end_freq-start_freq

            line_magnitude, = self.ax.plot(freqs, np.full(freqs.shape, magnitude), 'gainsboro')  # Plot magnitude
            line_upper_bound, = self.ax.plot(freqs, np.full(freqs.shape, magnitude + upper_bound), 'salmon')  # Plot upper bound
            line_lower_bound, = self.ax.plot(freqs, np.full(freqs.shape, magnitude - lower_bound), 'aqua')  # Plot lower bound

            self.draggable_lines_mag.append(DraggablePlotter(line_magnitude, self, 'gainsboro',midpoint))
            self.draggable_lines_upper.append(DraggablePlotter(line_upper_bound, self, 'salmon',midpoint))
            self.draggable_lines_lower.append(DraggablePlotter(line_lower_bound, self, 'aqua',midpoint))

        for draggable_line in self.draggable_lines_mag:
            draggable_line.connect()

        for draggable_line in self.draggable_lines_upper:
            draggable_line.connect()

        for draggable_line in self.draggable_lines_lower:
            draggable_line.connect()

        if self.app:
            self.app.canvas.draw()

    def set_data(self, filter_order):
        
        self.step = filter_order*16
        print("step is : ",self.step)
        self.xdata = np.linspace(0, 1, self.step)
        self.upper_ydata = np.full(self.xdata.shape, np.nan)
        self.lower_ydata = np.full(self.xdata.shape, np.nan)
         
        for draggable_line in self.draggable_lines_upper:
            x_data = draggable_line.get_xdata()
            y_data = draggable_line.get_ydata()

            # Interpolate y_data to match self.upper_xdata
            interpolated_y_data = np.interp(self.xdata, x_data, y_data, left=np.nan, right=np.nan)

            # Update self.upper_ydata with the interpolated values
            self.upper_ydata = np.where(np.isnan(self.upper_ydata), interpolated_y_data, self.upper_ydata)

        for draggable_line in self.draggable_lines_lower:
            x_data = draggable_line.get_xdata()
            y_data = draggable_line.get_ydata()

            # Interpolate y_data to match self.lower_xdata
            interpolated_y_data = np.interp(self.xdata, x_data, y_data, left=np.nan, right=np.nan)

            # Update self.lower_ydata with the interpolated values
            self.lower_ydata = np.where(np.isnan(self.lower_ydata), interpolated_y_data, self.lower_ydata)

        self.upper_ydata_lin=self.db_to_linear(self.upper_ydata)
        self.lower_ydata_lin=self.db_to_linear(self.lower_ydata)
        
        # Plot the updated upper_ydata
        self.ax.scatter(self.xdata, self.upper_ydata_lin, color='r', s=20, picker=5)
        # Plot the updated lower_ydata
        self.ax.scatter(self.xdata, self.lower_ydata_lin, color='b', s=20, picker=5)

        # Refresh the plot if the app is provided
        if self.app:
            self.app.canvas.draw()

    def get_upper_ydata(self):
        return self.upper_ydata

    def get_lower_ydata(self):
        return self.lower_ydata

    def get_xdata(self):
        return self.xdata  
    
    def db_to_linear(self,db_arr):
        # Create a mask for NaN values
        nan_mask = np.isnan(db_arr)

        # Apply the conversion to non-NaN values (magnitude)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)

        # Preserve NaN values
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def plot_result(self, result_coef):
        print("result plotter called")
        fir_coefficients = np.array(result_coef)
        print("Fir coef in mp",fir_coefficients)

        # Compute the FFT of the coefficients
        N = 5120  # Number of points for the FFT
        frequency_response = np.fft.fft(fir_coefficients, N)
        frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist


        # Compute the magnitude and phase response for positive frequencies
        magnitude_response = np.abs(frequency_response)[:N//2]

        # Convert magnitude response to dB
        magnitude_response_db = 20 * np.log10(np.where(magnitude_response == 0, 1e-10, magnitude_response))

        # print("magdb in mp",magnitude_response_db)

        # Normalize frequencies to range from 0 to 1
        normalized_frequencies = frequencies / np.max(frequencies)

        # Plot the updated upper_ydata
        self.ax.set_ylim([-0.5, 30])
        self.ax.plot(normalized_frequencies, magnitude_response, color='y')


        if self.app:
            self.app.canvas.draw()



    
    



# Example usage:
# Assuming app is defined and draggable_plotter.py is correctly implemented
# plotter = MagnitudePlotter(app)
# plotter.initiate_plot(data)  # Assuming data is defined
# plotter.get_upper_data()  # Call this method to update upper_ydata and plot it



    # def save_state(self):
    #     self.history.append((self.x.copy(), self.y.copy()))
    #     self.future.clear()

    # def smoothen_plot(self):
    #     kernel_size = self.app.kernel_slider.value()
    #     self.y = medfilt(self.y, kernel_size)
    #     self.save_state()
    #     self.redraw_plot()

    # def undo_plot(self):
    #     if len(self.history) > 1:
    #         self.future.append(self.history.pop())
    #         self.x, self.y = copy.deepcopy(self.history[-1])
    #         self.just_undone = True
    #         self.redraw_plot()

    # def redo_plot(self):
    #     if self.future:
    #         self.history.append(self.future.pop())
    #         self.x, self.y = copy.deepcopy(self.history[-1])
    #         self.just_undone = False
    #         self.redraw_plot()

    # def toggle_selection_mode(self):
    #     self.selection_mode = not self.selection_mode
    #     if self.selection_mode:
    #         self.app.selection_mode_button.setText("Exit Selection Mode")
    #     else:
    #         self.app.selection_mode_button.setText("Selection Mode")

    # def apply_flatten(self):
    #     if not self.selected_range:
    #         return
    #     start, end = self.selected_range
    #     start, end = sorted([start, end])
    #     target_y = self.app.flat_level_slider.value()
    #     indices = (self.x >= start) & (self.x <= end)
    #     self.y[indices] = target_y
    #     self.save_state()
    #     self.redraw_plot()
    #     if self.selection_rect:
    #         self.selection_rect.remove()
    #         self.selection_rect = None

    # def on_click(self, event):
    #     if event.inaxes != self.ax:
    #         return

    #     if self.selection_mode:
    #         self.selected_range = [event.xdata, None]
    #         self.selection_rect = Rectangle((event.xdata, self.ax.get_ylim()[0]), 0, self.ax.get_ylim()[1] - self.ax.get_ylim()[0], color='gray', alpha=0.3)
    #         self.ax.add_patch(self.selection_rect)
    #         self.app.canvas.draw()

    # def on_release(self, event):
    #     if self.selection_mode and self.selected_range:
    #         self.selected_range[1] = event.xdata
    #         self.selection_mode = False
    #         self.app.selection_mode_button.setText("Selection Mode")
    #         if self.selection_rect:
    #             self.selection_rect.set_width(self.selected_range[1] - self.selected_range[0])
    #             self.app.canvas.draw()

    # def on_motion(self, event):
    #     if event.inaxes != self.ax:
    #         return
    #     if self.selection_mode and self.selected_range:
    #         if self.selection_rect:
    #             self.selection_rect.set_width(event.xdata - self.selected_range[0])
    #             self.app.canvas.draw()

    # def update_draggable_point(self, event, move_only=False):
    #     x = event.xdata
    #     y = event.ydata
    #     if x is None or y is None:
    #         return
    #     if move_only:
    #         idx = np.argmin(np.abs(self.x - x))
    #         y = self.y[idx]
    #         self.draggable_point.set_data([self.x[idx]], [y])
    #     else:
    #         idx = np.argmin(np.abs(self.x - x))
    #         x = self.x[idx]
    #         self.draggable_point.set_data([x], [y])
    #         sigma = 1
    #         influence = np.exp(-0.5 * ((self.x - x) / sigma) ** 2)
    #         delta_y = y - self.y[idx]
    #         self.y += influence * delta_y
    #     self.redraw_plot()

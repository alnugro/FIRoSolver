import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy


try:
    from .draggable_plotter import DraggablePlotter
except:
    from draggable_plotter import DraggablePlotter


class MagnitudePlotter:
    def __init__(self ,day_night = False ,app=None):
        if day_night:
            plt.style.use('fivethirtyeight')
        else:
            plt.style.use('dark_background')

        self.app = app
        self.fig, self.ax = plt.subplots(figsize=(30, 4))
        self.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.98)
        
        plt.rcParams['lines.linewidth'] = 0.7
        self.day_night = day_night
        plt.rcParams['font.size'] = 11
        if self.day_night:
            self.ax.grid(True, linewidth=1)
            plt.rcParams['text.color'] = 'black'        # Set default color for all text
            plt.rcParams['axes.labelcolor'] = 'black'    # Set default color for axis labels
            plt.rcParams['xtick.color'] = 'black'       # Set default color for x-tick labels
            plt.rcParams['ytick.color'] = 'black'       # Set default color for y-tick labels
        else:
            self.ax.grid(True, linewidth=1)

        for spine in self.ax.spines.values():
            spine.set_linewidth(0.5)

        # Set DPI
        # plt.rcParams['figure.dpi'] = 20

        # Disable antialiasing globally
        plt.rcParams['lines.antialiased'] = False   # For lines in plots
        plt.rcParams['patch.antialiased'] = False   # For patches, like rectangles and polygons
        plt.rcParams['text.antialiased'] = False    # For text
        

        self.xdata_edges = []
        # var to put the object of Draggable plotter
        self.draggable_lines = None


    def initiate_plot(self, table_data, data_dict):

        #reset plot if its not empty
        if self.draggable_lines:
            print("clearing")
            self.draggable_lines.disconnect()
            self.draggable_lines = None

        # Clear the axes to remove all plot elements
        self.ax.cla()

        # self.ax.set_xlim([0, 1])
        # self.ax.set_ylim([-1, 1])
        if self.day_night:
            pass
        else:
            self.ax.grid()



        highest_upper=0
        lowest_lower=0
        # table_data = table_data[table_data[:, 3].argsort()]  # Sort table_data by the fourth column (start frequency)
        table_data = sorted(table_data, key=lambda x: x[4])
        
        print (f"table_data {table_data}")
        xdata = np.linspace(0, 1, 500)
        middle_y = np.full(xdata.shape, np.nan)
        upper_y = np.full(xdata.shape, np.nan)
        lower_y = np.full(xdata.shape, np.nan)
        print(f"table_data {table_data}")
        for row in table_data:
            magnitude = None
            type, gain, lower_bound, upper_bound, start_freq, end_freq = row
            if type == 'stopband':
                magnitude = 0
            elif type == 'passband':
                magnitude = 1
            else:
                raise ValueError(f"This type: {type} is not supported, only passband or stopband are supported.")
        
            magnitude *= gain

            lower_bound = 10 ** (lower_bound / 20)
            upper_bound = 10 ** (upper_bound / 20)
            
            self.xdata_edges.append(start_freq)
            self.xdata_edges.append(end_freq)

            # xdata_lower_index = np.searchsorted(xdata, start_freq)
            # xdata_upper_index = np.searchsorted(xdata, end_freq)

            if not(start_freq in xdata):
                xdata_lower_index = np.searchsorted(xdata, start_freq)
                xdata = np.insert(xdata, xdata_lower_index, start_freq)
                middle_y = np.insert(middle_y, xdata_lower_index, np.nan)
                upper_y = np.insert(upper_y, xdata_lower_index, np.nan)
                lower_y = np.insert(lower_y, xdata_lower_index, np.nan)
                
            if not(end_freq in xdata):
                xdata_upper_index = np.searchsorted(xdata, end_freq)
                xdata = np.insert(xdata, xdata_upper_index, end_freq)
                middle_y = np.insert(middle_y, xdata_upper_index, np.nan)
                upper_y = np.insert(upper_y, xdata_upper_index, np.nan)
                lower_y = np.insert(lower_y, xdata_upper_index, np.nan)
            
            start_index = np.where(xdata == start_freq)[0][0]
            end_index =  np.where(xdata == end_freq)[0][0]
            middle_y[start_index:end_index+1] = magnitude
            upper_y[start_index:end_index+1] = magnitude + upper_bound
            if type == 'passband':
                lower_y[start_index:end_index+1] = magnitude - lower_bound
            else:
                pass


                    
        self.draggable_lines = DraggablePlotter(self.fig, self.ax,self.app.canvas, self.day_night, self.app.position_label, self.app)
        self.update_plotter(data_dict, False)
        self.draggable_lines.initialize_plot(xdata, middle_y, upper_y, lower_y)




    def load_plot(self, input_dict, loaded_dict, history = None):

        #reset plot if its not empty
        if self.draggable_lines:
            print("clearing")
            self.draggable_lines.disconnect()
            self.draggable_lines = None

        # Clear the axes to remove all plot elements
        self.ax.cla()

        


        xdata = np.array(loaded_dict['xdata'])
        middle_y = np.array(loaded_dict['middle_y'])
        upper_y = np.array(loaded_dict['upper_y'])
        lower_y = np.array(loaded_dict['lower_y'])

        
        
        self.draggable_lines = DraggablePlotter(self.fig, self.ax,self.app.canvas, self.day_night, self.app.position_label, self.app)
        self.update_plotter(input_dict, False)
        self.draggable_lines.initialize_plot(xdata, middle_y, upper_y, lower_y)
        if history:
            self.draggable_lines.set_history(history)



    def update_plotter(self, data_dict, redraw = True):
        if self.draggable_lines:
            self.draggable_lines.update_plotter_data(data_dict, redraw)
    
    def undo_plot(self):
        if self.draggable_lines:
            self.draggable_lines.undo_plot()
    
    def redo_plot(self):
        if self.draggable_lines:
            self.draggable_lines.redo_plot()

    def smoothen_plot(self):
        if self.draggable_lines:
            self.draggable_lines.smoothen_plot()

    def apply_flatten(self):
        if self.draggable_lines:
            self.draggable_lines.apply_flatten()
        
    def delete_selected(self):
        if self.draggable_lines:
            self.draggable_lines.delete_selected()
    
    def mirror_upper_to_lower(self):
        if self.draggable_lines:
            self.draggable_lines.mirror_upper_to_lower()
    
    def mirror_lower_to_upper(self):
        if self.draggable_lines:
            self.draggable_lines.mirror_lower_to_upper()

    def nyquist_to_sampling(self):
        if self.draggable_lines:
            ret = self.draggable_lines.nyquist_to_sampling()
            return ret
        else: return 1
    
    def sampling_to_nyquist(self):
        if self.draggable_lines:
            ret = self.draggable_lines.sampling_to_nyquist()
            return ret
        else: return 1
        
    def reset_draggable_points(self):
        if self.draggable_lines:
            ret = self.draggable_lines.reset_draggable_points()
            return ret
        else: return 1
        
 
    def get_frequency_bounds(self, only_plot = False):
        if not(self.draggable_lines):
            return None, None, None, None
        
        #array to save the transition band
        cutoffs_upper_ydata = []
        cutoffs_lower_ydata = []

        cutoffs_x = copy.deepcopy(self.xdata_edges)

        xdata , middle_y , lower_y, upper_y = self.draggable_lines.get_plot_data()

        if only_plot:
            return xdata , middle_y , lower_y, upper_y
        
        upper_ydata = copy.deepcopy(upper_y)
        lower_ydata = copy.deepcopy(lower_y)
        
        for xdata_index, xdata_value in enumerate(xdata):
            if np.isnan(lower_ydata[xdata_index]) and ~np.isnan(upper_ydata[xdata_index]):
                lower_ydata[xdata_index] = 0
        
        for xcut in cutoffs_x:
            index = np.where(xdata == xcut)[0][0]
            cutoffs_upper_ydata.append(upper_ydata[index])
            cutoffs_lower_ydata.append(lower_ydata[index])
        
               

        return xdata, upper_ydata, lower_ydata ,cutoffs_x, cutoffs_upper_ydata, cutoffs_lower_ydata

    def get_current_data(self):
        x, middle_y, lower_y, upper_y = [], [], [], []
        history = []

        # Reset plot if it's not empty
        if self.draggable_lines:
            x, middle_y, lower_y, upper_y = self.draggable_lines.get_plot_data()
            history = self.draggable_lines.get_history()
            print("clearing")
            self.draggable_lines.disconnect()
            self.draggable_lines = None
            self.ax.cla()  # Clear the axis
        else:
            return None
        
        return x, middle_y, lower_y, upper_y , history
        

    
    
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


    def result_plotter(self, result):

        pass


        





    
    



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

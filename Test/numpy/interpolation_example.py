import numpy as np

# Example data
xdata = np.array([1.5 , 2.5, 3, 4.5, 5])
current_xdata = np.array([1, 2, 3, 4, 5])  # x-data with NaN
current_ydata = np.array([10, 20, np.nan, 40, 50])  # y-data with NaN

# Interpolation
interpolated_current_ydata = np.interp(xdata, current_xdata, current_ydata, left=np.nan, right=np.nan)

print(interpolated_current_ydata)
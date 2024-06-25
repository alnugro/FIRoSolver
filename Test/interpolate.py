import numpy as np
import matplotlib.pyplot as plt

# Original data points
xp = np.array([0, 1, 2, 5])
fp = np.array([0, 1, 4, 25])

# Points to interpolate
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

# Interpolated values
interpolated_values = np.interp(x, xp, fp)

print("Interpolated values:", interpolated_values)

# Plotting for visualization
plt.plot(xp, fp, 'o', label='Original data points')
plt.plot(x, interpolated_values, 'x', label='Interpolated points')
plt.legend()
plt.show()

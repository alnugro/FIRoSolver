import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the original function
def original_function(x, y):
    return x / y

# Define piecewise linear function
def piecewise_linear(X, a1, b1, c1, a2, b2, c2):
    x, y = X
    r = x / y
    return np.piecewise(r, 
                        [r <= 1, r > 1], 
                        [lambda r: a1 * r + b1 * y + c1, 
                         lambda r: a2 * r + b2 * y + c2])

# Generate data points for fitting
x = np.linspace(0.1, 10, 100)
y = np.linspace(0.1, 10, 100)
x, y = np.meshgrid(x, y)
z = original_function(x, y)

# Flatten the arrays for curve fitting
x_data = x.ravel()
y_data = y.ravel()
z_data = z.ravel()

# Define initial guesses for the parameters
initial_guesses = [1, 0, 0, 1, 0, 0]

# Fit the piecewise linear function to the data
params, _ = curve_fit(piecewise_linear, (x_data, y_data), z_data, p0=initial_guesses)

# Extract fitted parameters
a1, b1, c1, a2, b2, c2 = params

# Define the fitted piecewise linear function
def fitted_piecewise_linear(x, y):
    return piecewise_linear((x, y), a1, b1, c1, a2, b2, c2)

# Plot the original and fitted functions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', label='Original')
ax.scatter(x, y, fitted_piecewise_linear(x, y), color='red', label='Piecewise Linear Approximation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

import numpy as np

# Example of non-linear distortion
def non_linear_distortion(x, alpha, beta):
    return x + alpha * x**2 + beta * x**3

# Example inverse filter
def inverse_filter(y, alpha, beta, iterations=5):
    x = y.copy()
    for _ in range(iterations):
        x = y - alpha * x**2 - beta * x**3
    return x

# Generate a test signal
fs = 44100  # Sampling frequency
t = np.linspace(0, 1, fs)
x_original = np.sin(2 * np.pi * 440 * t)  # A 440 Hz sine wave

# Apply non-linear distortion
alpha, beta = 0.5, 0.2
y_distorted = non_linear_distortion(x_original, alpha, beta)

# Apply inverse filter to correct distortion
x_corrected = inverse_filter(y_distorted, alpha, beta)

# Compare original and corrected signals
import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, x_original, label='Original Signal')
plt.plot(t, y_distorted, label='Distorted Signal')
plt.plot(t, x_corrected, label='Corrected Signal')
plt.legend()
plt.show()

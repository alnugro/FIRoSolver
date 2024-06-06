import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.fft import fft, fftfreq

# Define the desired frequency response
def desired_magnitude_phase(omega):
    magnitude = np.piecewise(omega, 
                             [omega < 0.2 * np.pi, (omega >= 0.2 * np.pi) & (omega <= 0.4 * np.pi), omega > 0.4 * np.pi],
                             [0, 1, 0])  # Bandpass magnitude response
    phase = np.piecewise(omega, 
                         [omega < 0.2 * np.pi, (omega >= 0.2 * np.pi) & (omega <= 0.4 * np.pi), omega > 0.4 * np.pi],
                         [0, np.pi / 4, 0])  # Nonlinear phase response
    return magnitude, phase

# FIR filter design parameters
N = 50  # Number of filter coefficients
M = 500  # Number of frequency samples

# Sample points
omega = np.linspace(0, np.pi, M)
magnitude, phase = desired_magnitude_phase(omega)

for i in magnitude:
    print(i)
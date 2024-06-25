import numpy as np
import matplotlib.pyplot as plt

# Define your FIR filter coefficients
h = [
    0, -2, -4, -4, -1, -1, -4, -4, -2, 0
]
fir_coefficients = np.array(h)  # Example coefficients

# Compute the FFT of the coefficients
N = 512  # Number of points for the FFT
frequency_response = np.fft.fft(fir_coefficients, N)
frequencies = np.fft.fftfreq(N, d=1.0)[:N//2]  # Extract positive frequencies up to Nyquist

# Compute the magnitude and phase response for positive frequencies
magnitude_response = np.abs(frequency_response)[:N//2]
phase_response = np.angle(frequency_response)[:N//2]

# Normalize frequencies to range from 0 to 1
normalized_frequencies = frequencies / np.max(frequencies)

# Plot the magnitude response
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(normalized_frequencies, magnitude_response)
plt.title('Magnitude Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.grid()

# Plot the phase response
plt.subplot(2, 1, 2)
plt.plot(normalized_frequencies, phase_response)
plt.title('Phase Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Phase (radians)')
plt.grid()

plt.tight_layout()
plt.show()

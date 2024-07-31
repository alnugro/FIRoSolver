import numpy as np
import matplotlib.pyplot as plt

# Define your FIR filter coefficients
h = [
  -0.02010411882885732,
  -0.05842798004352509,
  -0.061178403647821976,
  -0.010939393385338943,
  0.05125096443534972,
  0.033220867678947885,
  -0.05655276971833928,
  -0.08565500737264514,
  0.0633795996605449,
  0.31085440365663597,
  0.4344309124179415,
  0.31085440365663597,
  0.0633795996605449,
  -0.08565500737264514,
  -0.05655276971833928,
  0.033220867678947885,
  0.05125096443534972,
  -0.010939393385338943,
  -0.061178403647821976,
  -0.05842798004352509,
  -0.02010411882885732

]
fir_coefficients = np.array(h)  # Example coefficients

# Compute the FFT of the coefficients
N = 1000  # Number of points for the FFT
frequency_response = np.fft.fft(fir_coefficients, N)
frequencies = np.fft.fftfreq(N, d=1.0)  # Extract positive frequencies up to Nyquist

print(frequencies)

# Compute the magnitude and phase response for positive frequencies
magnitude_response = np.abs(frequency_response)
# phase_response = np.angle(frequency_response)[:N//2]

# # Normalize frequencies to range from 0 to 1
# normalized_frequencies = frequencies / np.max(frequencies)

# Plot the magnitude response
plt.figure(figsize=(12, 6))
plt.plot(frequencies, magnitude_response)
plt.title('Magnitude Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.grid()


plt.tight_layout()
plt.show()

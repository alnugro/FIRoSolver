import numpy as np
import matplotlib.pyplot as plt


def rework_list(arr):
    arr_np = np.array(arr)
    if len(arr_np) % 2 == 0:
        # Even length: reverse the array excluding the first element (so no repeat of 0) and append the original array
        result = np.concatenate((arr_np[::-1], arr_np[1:]))
    else:
        # Odd length: reverse the array and append the original array
        result = np.concatenate((arr_np[::-1], arr_np))
    return result


# Define your FIR filter coefficients
h = [
  472,
  248,
  0,
  96,
  42,
  32,
  36,
  0,
  21,
  8,
  6,
  6
]
fir_coefficients_np = np.array(h)  # Example coefficients
fir_coefficients_word = [coef/2**10 for coef in fir_coefficients_np]
fir_coefficients = rework_list(fir_coefficients_word)

# Compute the FFT of the coefficients
N = 1000  # Number of points for the FFT
frequency_response = np.fft.fft(fir_coefficients, N)
frequencies= np.fft.fftfreq(N, d=1.0)  # Extract positive frequencies up to Nyquist


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

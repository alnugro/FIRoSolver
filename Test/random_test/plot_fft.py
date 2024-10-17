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
  0.5, 0.25, 0.0, -0.125
]
fir_coefficients_np = np.array(h)  # Example coefficients
fir_coefficients_word = [coef for coef in fir_coefficients_np]
fir_coefficients = rework_list(fir_coefficients_word)

# Compute the FFT of the coefficients
N = 100  # Number of points for the FFT
frequency_response = np.fft.fft(fir_coefficients, N)
frequencies= np.fft.fftfreq(N, d=0.5)  # Extract positive frequencies up to Nyquist




# Compute the magnitude and phase response for positive frequencies
magnitude_response = np.abs(frequency_response)
# phase_response = np.angle(frequency_response)[:N//2]

# # Normalize frequencies to range from 0 to 1
# normalized_frequencies = frequencies / np.max(frequencies)

pair = []
for i in range(50):
    pair.append((np.array(frequencies[i]).tolist(), np.array(magnitude_response[i]).tolist()))
print(frequencies[:50])
print(magnitude_response[:50])
print(pair)
print(fir_coefficients)

# Plot the magnitude response
plt.figure(figsize=(12, 6))
plt.plot(frequencies, magnitude_response)
plt.title('Magnitude Response')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude')
plt.grid()


plt.tight_layout()
plt.show()

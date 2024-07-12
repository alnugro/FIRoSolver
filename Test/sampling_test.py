import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000.0  # Sampling frequency in Hz
T = 1.0  # Total duration in seconds
N = 512  # Number of points for the FFT

# Generate time-domain signal (composite of multiple sine waves)
t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector
signal = (np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
          np.sin(2 * np.pi * 120 * t) +  # 120 Hz component
          np.sin(2 * np.pi * 300 * t))  # 300 Hz component

# Sample the signal
sampled_signal = signal[:N]

# Compute the FFT
frequency_response = np.fft.fft(sampled_signal, N)

# Calculate the DFT sample frequencies
frequencies = np.fft.fftfreq(N, d=1.0/fs)

# Extract positive frequencies up to the Nyquist frequency
positive_frequencies = frequencies[:N//2]

# Plot the time-domain signal
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:N], sampled_signal)
plt.title('Time-Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the magnitude of the frequency response (without normalization)
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies, np.abs(frequency_response[:N//2]))
plt.title('Frequency-Domain Signal (Up to Nyquist Frequency)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

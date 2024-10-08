import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.fft import fft, fftfreq

# Define the desired frequency response
def desired_magnitude_phase(omega):
    magnitude = np.piecewise(omega, 
                             [omega < 0.2 * np.pi, (omega >= 0.2 * np.pi) & (omega <= 0.4 * np.pi), omega > 0.4 * np.pi],
                             [0.001, 1, 0.001])  # Bandpass magnitude response
    phase = np.piecewise(omega, 
                         [omega < 0.2 * np.pi, (omega >= 0.2 * np.pi) & (omega <= 0.4 * np.pi), omega > 0.4 * np.pi],
                         [0, np.pi / 4, 0])  # Nonlinear phase response
    return magnitude, phase

# Compute the group delay
def group_delay(phase, omega):
    unwrapped_phase = np.unwrap(phase)
    return -np.gradient(unwrapped_phase, omega)

# FIR filter design parameters
N = 3000  # Number of filter coefficients
M = 500  # Number of frequency samples

# Sample points
omega = np.linspace(0, np.pi, M)
magnitude, phase = desired_magnitude_phase(omega)

# Desired complex response
H_d_samples = magnitude * np.exp(1j * phase)

# Construct the matrix for the least squares problem
A_real = np.zeros((M, N))
A_imag = np.zeros((M, N))

for k in range(M):
    for n in range(N):
        A_real[k, n] = np.cos(omega[k] * n)
        A_imag[k, n] = np.sin(omega[k] * n)

# Desired response vector
b_real = H_d_samples.real
b_imag = H_d_samples.imag

# Solve the least squares problem separately for real and imaginary parts
h_real, _, _, _ = lstsq(A_real, b_real)
h_imag, _, _, _ = lstsq(A_imag, b_imag)

# Combine the real and imaginary parts to get the final filter coefficients
h = h_real + 1j * h_imag

# Ensure coefficients are purely real
h = h.real

# Perform FFT to obtain the frequency response of the designed filter
H = fft(h, n=1024)
freqs = fftfreq(1024, d=1)[:512]

# Normalize frequency
normalized_freqs = freqs / max(freqs)

# Desired response for comparison
omega_full = np.linspace(0, np.pi, 512)
H_d_full_magnitude, H_d_full_phase = desired_magnitude_phase(omega_full)
H_d_full = H_d_full_magnitude * np.exp(1j * H_d_full_phase)

# Compute group delay
group_delay_desired = group_delay(np.angle(H_d_full), omega_full)
group_delay_designed = group_delay(np.angle(H[:512]), normalized_freqs * np.pi)

# Plotting
plt.figure(figsize=(12, 8))

# Magnitude response in dB
plt.subplot(2, 1, 1)
plt.plot(omega_full / np.pi, 20 * np.log10(np.abs(H_d_full)), label='Desired Magnitude Response (dB)', color='blue')
plt.plot(normalized_freqs, 20 * np.log10(np.abs(H[:512])), label='Designed Filter Magnitude Response (dB)', color='red', linestyle='--')
plt.title('Magnitude Response')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.legend()

# Group delay response
plt.subplot(2, 1, 2)
plt.plot(omega_full / np.pi, group_delay_desired, label='Desired Group Delay', color='blue')
plt.plot(normalized_freqs, group_delay_designed, label='Designed Filter Group Delay', color='red', linestyle='--')
plt.title('Group Delay Response')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Group Delay (samples)')
plt.legend()

plt.tight_layout()
plt.show()

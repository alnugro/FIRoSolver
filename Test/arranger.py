import numpy as np
import matplotlib.pyplot as plt

# Create example arrays for demonstration
x_a = np.arange(0, 1.0001, 0.0001)
y_a = np.sin(2 * np.pi * x_a)  # Example function, replace with actual values
y_a[500:10000] = np.nan  # Introduce some NaNs for testing

x_b = np.array([0.00005, 0.0023, 0.4567, 0.7890])  # Example x-values for b
y_b = np.zeros_like(x_b)

# Function to find the nearest non-NaN values
def find_nearest(arr, value):
    idx = (np.abs(arr - value)).argmin()
    if np.isnan(arr[idx]):
        left_idx = right_idx = idx
        while left_idx > 0 and np.isnan(arr[left_idx]):
            left_idx -= 1
        while right_idx < len(arr) - 1 and np.isnan(arr[right_idx]):
            right_idx += 1
        if np.isnan(arr[left_idx]):
            left_idx = None
        if np.isnan(arr[right_idx]):
            right_idx = None
        return left_idx, right_idx
    else:
        return idx, idx

# Approximate y_b values based on x_a and y_a
for i, xb in enumerate(x_b):
    exact_match_indices = np.where(x_a == xb)[0]
    if exact_match_indices.size > 0 and not np.isnan(y_a[exact_match_indices[0]]):
        y_b[i] = y_a[exact_match_indices[0]]
    else:
        left_idx, right_idx = find_nearest(x_a, xb)
        if left_idx is not None and right_idx is not None:
            if left_idx == right_idx:
                y_b[i] = y_a[left_idx]
            else:
                y_b[i] = (y_a[left_idx] + y_a[right_idx]) / 2
        elif left_idx is not None:
            y_b[i] = y_a[left_idx]
        elif right_idx is not None:
            y_b[i] = y_a[right_idx]
        else:
            y_b[i] = np.nan  # If both sides are NaN, set NaN

# Print results
for xb, yb in zip(x_b, y_b):
    print(f"x_b: {xb}, y_b: {yb}")

# Plot the results using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(x_a, y_a, label='Array a', alpha=0.5)
plt.scatter(x_b, y_b, color='red', label='Array b', zorder=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

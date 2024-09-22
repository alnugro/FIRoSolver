import numpy as np

# Your initial array
initial_array = np.array([0, 0.2, 0.6, 0.9, 1])

# Create the array from 0 to 1 with a step of 0.03
new_array = np.arange(0, 1.03, 0.03)  # 1.03 to ensure 1 is included

# Combine both arrays
combined_array = np.concatenate((initial_array, new_array))

# Remove duplicates and sort the result
combined_array = np.unique(combined_array)

print(combined_array)

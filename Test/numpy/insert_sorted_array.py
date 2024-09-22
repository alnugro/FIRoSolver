import numpy as np

# Your sorted array
sorted_array = np.array([0, 1, 2, 5, 6])

# The value you want to insert
value_to_insert = 3

# Find the index where the value should be inserted
index = np.searchsorted(sorted_array, value_to_insert)

# Insert the value at the found index
sorted_array = np.insert(sorted_array, index, value_to_insert)

print(sorted_array)  # Output: [0 1 2 3 5 6]

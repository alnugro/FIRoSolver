import numpy as np

def interpolate_and_insert(arr, index1, index2, num_values=1):
    # Get values at the two indices
    value1 = arr[index1]
    value2 = arr[index2]
    
    # Create the interpolated values
    interpolated_values = np.linspace(value1, value2, num=num_values + 2)[1:-1]  # Exclude the endpoints
    
    # Insert the interpolated values
    new_arr = np.insert(arr, index2, interpolated_values)
    
    return new_arr

# Example usage:
my_array = np.array([1, 2.5, 4, 8])
index1 = 1  # Value at index 1 is 2
index2 = 2  # Value at index 2 is 4
num_values = 3  # Number of values to insert

new_array = interpolate_and_insert(my_array, index1, index2, num_values)
print(new_array)

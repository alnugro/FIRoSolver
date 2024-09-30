import numpy as np

def find_closest_below_numpy(arr, target):
    # Convert to a NumPy array if it's not already
    arr = np.array(arr)
    
    # Filter values that are less than or equal to the target
    filtered_arr = arr[arr <= target]
    
    # If no value is found, return None or NaN
    if filtered_arr.size == 0:
        return None
    
    # Find the closest value
    return filtered_arr[np.abs(filtered_arr - target).argmin()]

# Example usage:
my_list = [1.2, 2.5, 5.8, 8.3]
target_value = 4.0

closest_value = find_closest_below_numpy(my_list, target_value)
print(f"The closest value to {target_value} is {closest_value}")

space = 15
# Initialize freq_upper and freq_lower with NaN values
freqx_axis = np.linspace(0, 1, space)
print(freqx_axis)
print(1/len(freqx_axis))
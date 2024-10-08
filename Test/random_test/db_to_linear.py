import numpy as np

def db_to_linear(db_array):
    """
    Converts a NumPy array from dB to linear scale, preserving NaNs.

    Parameters:
    db_array (np.ndarray): The input array in dB.

    Returns:
    np.ndarray: The array converted to linear scale, with NaNs preserved.
    """
    # Create a mask for NaN values
    nan_mask = np.isnan(db_array)

    # Apply the conversion to non-NaN values (magnitude)
    linear_array = np.zeros_like(db_array)
    linear_array[~nan_mask] = 10 ** (db_array[~nan_mask] / 20)

    # Preserve NaN values
    linear_array[nan_mask] = np.nan

    return linear_array

# Example usage
db_array = np.array([20, 30, np.nan, 40, 50])
linear_array = db_to_linear(db_array)

print("dB array:", db_array)
print("Linear array:", linear_array)

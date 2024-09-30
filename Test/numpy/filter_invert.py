import numpy as np

def rework_list(arr):
    arr_np = np.array(arr)
    if len(arr_np) % 2 == 0:
        # Even length: reverse the array excluding the first element (so no repeat of 0) and append the original array
        result = np.concatenate((arr_np[::-1], arr_np[1:]))
    else:
        # Odd length: reverse the array and append the original array
        result = np.concatenate((arr_np[::-1], arr_np))
    return result

# Example usage
arr_even = ['0', 1, 2, 3, 4, 5]
arr_odd = [0, 1, 2, 3, 4]

result_even = rework_list(arr_even)
result_odd = rework_list(arr_odd)

print("Even length result:", result_even)
print("Odd length result:", result_odd)

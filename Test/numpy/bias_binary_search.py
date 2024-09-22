def find_max_zero_sum_bias(sat_list):
    low, high = 0, len(sat_list) - 1
    max_zero = -1  # Default value if no 'sat' is found

    while low <= high:
        # Bias the mid calculation toward the lower half
        mid = low + (high - low) // 3  # Bias towards lower half
        print(f"Checking mid index {mid}: {sat_list[mid]}")

        if sat_list[mid] == 'sat':
            max_zero = mid  # Update max_zero to the current 'sat' index
            low = mid + 1   # Move to the right half to find a higher 'sat'
        else:
            high = mid - 1  # Move to the left half since 'unsat' was found

    print(f"Max zero sum found at index: {max_zero}")
    return max_zero

def find_max_zero_sum(sat_list):
    low, high = 0, len(sat_list) - 1
    max_zero = -1  # Default value if no 'sat' is found

    while low <= high:
        mid = (low + high) // 2
        print(f"Checking mid index {mid}: {sat_list[mid]}")

        if sat_list[mid] == 'sat':
            max_zero = mid  # Update max_zero to the current 'sat' index
            low = mid + 1   # Move to the right half to find a higher 'sat'
        else:
            high = mid - 1  # Move to the left half since 'unsat' was found

    print(f"Max zero sum found at index: {max_zero}")
    return max_zero


# Example usage with a larger list
sat_list = ['sat'] * 3 + ['unsat'] * 30  # 6 'sat' values followed by 14 'unsat' values

max_zero_sum = find_max_zero_sum_bias(sat_list)
print(f"Max zero sum index bias: {max_zero_sum}\n")

max_zero_sum = find_max_zero_sum(sat_list)
print(f"\nMax zero sum index: {max_zero_sum}")


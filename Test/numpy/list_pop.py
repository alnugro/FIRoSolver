def remove_trailing_zeros(lst):
    while lst and lst[-1] == 0:
        lst.pop()
    return lst

# Examples:
print(remove_trailing_zeros([1, 3, 1, 4, 0, 0]))  # Output: [1, 3, 1, 4]
print(remove_trailing_zeros([0, 0, 0]))  # Output: [1, 3, 1, 4]
print(remove_trailing_zeros([0, 1, 5, 1, 0]))     # Output: [0, 1, 5, 1]

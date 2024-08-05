import random

def generate_randomized_lits_and_weights(num_lists_start, num_lists_end, list_length_start, list_length_end, weight_min, weight_max):
    """
    Generate a list of lists with consecutive integers and a corresponding list of weights.
    The number of sublists and their lengths are chosen randomly within specified ranges.
    The weights list has the same length as the number of sublists, with random values within specified bounds.
    
    Args:
    - num_lists_start (int): Minimum number of sublists.
    - num_lists_end (int): Maximum number of sublists.
    - list_length_start (int): Minimum length of the sublists.
    - list_length_end (int): Maximum length of the sublists.
    - weight_min (int/float): Minimum value for weights.
    - weight_max (int/float): Maximum value for weights.
    
    Returns:
    - lits (list of list of int): List of lists with consecutive integers.
    - weights (list of int/float): List of weights corresponding to the sublists in lits.
    """
    num_lists = random.randint(num_lists_start, num_lists_end)
    sublist_length = random.randint(list_length_start, list_length_end)
    
    lits = []
    current_value = 1
    
    for _ in range(num_lists):
        sublist = [current_value + i for i in range(sublist_length)]
        lits.append(sublist)
        current_value += sublist_length
    
    weights = [random.uniform(weight_min, weight_max) for _ in range(num_lists)]
    
    return lits, weights

# Example usage:
num_lists_start = 3
num_lists_end = 5
list_length_start = 4
list_length_end = 6
weight_min = -5
weight_max = 5

random_lits, random_weights = generate_randomized_lits_and_weights(num_lists_start, num_lists_end, list_length_start, list_length_end, weight_min, weight_max)
print("Lits:", random_lits)
print("Weights:", random_weights)

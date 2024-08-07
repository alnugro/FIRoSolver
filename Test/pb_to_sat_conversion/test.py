import random
import numpy as np
from pb2cnf import PB2CNF
from pysat.solvers import Solver
from rat2bool import Rat2bool
import time

def calculate_result(model, lits, weights, fracW):
    """Calculate the weighted sum based on the model using 2's complement representation."""
    result = 0
    for i, weight in enumerate(weights):
        for j in range(len(lits[i])):
            value = model[lits[i][j] - 1]

            bool_weight = 2**(-fracW+j)
            if j == len(lits[i])-1:
                bool_weight = -2**(-fracW+j)
            if value > 0:
                result += bool_weight*weight
    return result

def test_pb2cnf(case, lits, weights, bound, fracW, top_var):
    global iter
    iter+=1
    pb = PB2CNF(top_var=top_var)
    if case == 'atleast':
        cnf = pb.atleast(weights, lits, bound, fracW)
    elif case == 'atmost':
        cnf = pb.atmost(weights, lits, bound, fracW)
    elif case == 'equals':
        cnf = pb.equal(weights, lits, bound, fracW)
    else:
        raise ValueError("Unknown case type")

    
    solver = Solver(name='Cadical195')
    for clause in cnf:
        solver.add_clause(clause)
    failed_test = []

    

    print(f"Solver Running {iter}")
    is_sat = solver.solve()
    if is_sat:
        model = solver.get_model()
        print("\nSAT")
        
        result = calculate_result(model, lits, weights, fracW)
        rounded_result = round(result, fracW)
        print(f"case : {case}")

        print(f"Calculated Result: {rounded_result}")

        print(f"Original Bound Value: {bound}")

        if case == 'atleast':
            if rounded_result < bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits} ,{weights}"])
                print("\n*****************atleast failed*****************\n")
        elif case == 'atmost':
            if rounded_result > bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits},{weights}"])
                print("\n*****************atmost failed*****************\n")
        elif case == 'equals':
            if rounded_result > bound or rounded_result < bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits},{weights}"])
                print("\n*****************equals failed*****************\n")
        print("Model verification passed")
    else:
        print("UNSAT")
        
    solver.delete()
    return failed_test

def generate_randomized_lits(num_lists_start, num_lists_end, list_length_start, list_length_end):

    num_lists = random.randint(num_lists_start, num_lists_end)
    sublist_length = random.randint(list_length_start, list_length_end)
    
    lits = []
    current_value = 1
    
    for _ in range(num_lists):
        sublist = [current_value + i for i in range(sublist_length)]
        lits.append(sublist)
        current_value += sublist_length
    
    
    
    return lits

iter = 0

# Example usage:

num_lists_start = 10
num_lists_end = 15
list_length_start = 10
list_length_end = 15


# Set a random seed for reproducibility
random_seed = 1
random.seed(random_seed)
rat = Rat2bool()
    
start_time= time.time()

total_test = 100 # Example number of tests, adjust as needed
failed_test = []

for i in range(total_test):
    random_lits = generate_randomized_lits(num_lists_start, num_lists_end, list_length_start, list_length_end)
    fracW_min = 4
    fracW_max = len(random_lits[0]) - 5
    fracW = random.randint(fracW_min, fracW_max)
    bound_min = (-2**(len(random_lits[0])-fracW-1))
    bound_max = (2**(len(random_lits[0])-fracW-1))
    bound = random.uniform(bound_min, bound_max)

    top_var = max(max(lit_group) for lit_group in random_lits)
    weight_min = -2**(len(random_lits[0])-fracW-1)
    weight_max = 2**(len(random_lits[0])-fracW-1)

    random_weights = [random.uniform(weight_min, weight_max) for _ in range(len(random_lits))]

    # round the inputs
    bound_round = rat.frac2round([bound],len(random_lits[0]), fracW)
    bound = bound_round[0]
    random_weights = rat.frac2round(random_weights,len(random_lits[0]), fracW)

    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")
    
    failed_test += test_pb2cnf('atleast', random_lits, random_weights, bound, fracW, top_var)

    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")


    failed_test += test_pb2cnf('atmost', random_lits, random_weights, bound, fracW, top_var)

    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")


    failed_test += test_pb2cnf('equals', random_lits, random_weights, bound, fracW, top_var)

failed_test = np.array(failed_test)
end_time = time.time()
print(f"Total failed tests: {len(failed_test)}")
print(f"failed tests description: \n{failed_test}")
print(f"Total tests done: {total_test*3}")
print(f"Duration: {end_time-start_time}")





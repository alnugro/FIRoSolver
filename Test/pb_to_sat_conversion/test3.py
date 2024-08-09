import random
from pb2cnf import PB2CNF
from pysat.solvers import Solver
from rat2bool import Rat2bool
import numpy as np
import time

def calculate_result(model, lits, weights, fracW):
    """Calculate the weighted sum based on the model using scaled integers."""
    result = 0 
    scale_factor = 2 ** fracW  # Scale factor to avoid floating-point arithmetic

    for k, weight in enumerate(weights):
        for j in range(len(lits[k])):
            weight_int = int(weight)
            value = model[lits[k][j] - 1]
            # print("val", value)

            # Calculate bool_weight as an integer by scaling
            bool_weight = 2 ** j
            if j == len(lits[k]) - 1:
                bool_weight = -bool_weight

            if value > 0:
                # Perform multiplication entirely with integers
                result_temp = int(bool_weight * weight_int)
                result += result_temp

            # print("wei", weight)
            # print("bow", bool_weight)
            # print("res", result)

    # Scale down the result by dividing by the scale factor
    final_result = result // scale_factor
    return final_result

def test_pb2cnf(case, lits, weights, bound, fracW, top_var):
    global iter
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
        max_cnf = max(max(cnf_group) for cnf_group in cnf)

        print(model)
        print("\nSAT")
        
        result = calculate_result(model, lits, weights, fracW)
        rounded_result = round(result, fracW)
        print(f"case : {case}")

        print(f"Calculated Result: {rounded_result}")

        print(f"Original Bound Value: {bound}")

        if case == 'atleast':
            if rounded_result < bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits} ,{weights}, iter: {iter}"])
                print("\n*****************atleast failed*****************\n")
        elif case == 'atmost':
            if rounded_result > bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits},{weights}, iter: {iter}"])
                print("\n*****************atmost failed*****************\n")
        elif case == 'equals':
            if rounded_result > bound or rounded_result < bound:
                failed_test.append([f"+++++++Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}, {lits},{weights}, iter: {iter}"])
                print("\n*****************equals failed*****************\n")
        print("Model verification passed")
    else:
        print("UNSAT")
        max_cnf = 0
        
    solver.delete()
    return failed_test, max_cnf
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




# Set a random seed for reproducibility
random_seed = 1
random.seed(random_seed)
rat = Rat2bool()
    


total_test = 50 # Example number of tests, adjust as needed
failed_test = []
result_test = []

for i in range(total_test):
    iter+=1

    start_time= time.time()
    num_lists_start = 10 
    num_lists_end = 15
    list_length_start = 10
    list_length_end = 20

    random_lits = generate_randomized_lits(num_lists_start, num_lists_end, list_length_start, list_length_end)
    fracW_min = 2
    fracW_max = len(random_lits[0]) - 2
    fracW = random.randint(fracW_min, fracW_max)
    bound_min = (-2**(len(random_lits[0])-fracW-1)+1)
    bound_max = (2**(len(random_lits[0])-fracW-1)-1)
    bound = random.uniform(bound_min, bound_max)

    weight_min = -2**(len(random_lits[0])-fracW-1)+1
    weight_max = 2**(len(random_lits[0])-fracW-1)-1

    random_weights = [random.uniform(weight_min, weight_max) or random.uniform(1, weight_max) for _ in range(len(random_lits))]

    print("before",random_weights)
    # round the inputs
    bound_round = rat.frac2round([bound],len(random_lits[0]), fracW)
    bound = bound_round[0]
    random_weights = rat.frac2round(random_weights,len(random_lits[0]), fracW)
    print("after",random_weights)

    # random_weights = [134002]
    # random_lits = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    # fracW = 4
    # bound = 133246
    top_var = max(max(lit_group) for lit_group in random_lits)



    max_cnf_res = 0
    

    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")
    
    failed_test_temp, max_cnf= test_pb2cnf('atleast', random_lits, random_weights, bound, fracW, top_var)
    max_cnf_res+=max_cnf
    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")


    failed_test_temp, max_cnf= test_pb2cnf('atmost', random_lits, random_weights, bound, fracW, top_var)
    max_cnf_res+=max_cnf

    print(f"\n\n--------Start Test----------- {i + 1}:")
    print(f"Top Var: {top_var}")
    print(f"Lits: {random_lits}")
    print(f"Weights: {random_weights}")
    print(f"Bound: {bound}")
    print(f"Fractional Width (fracW): {fracW}")


    failed_test_temp, max_cnf= test_pb2cnf('equals', random_lits, random_weights, bound, fracW, top_var)
    max_cnf_res+=max_cnf

    end_time = time.time()
    result_test.append([f"duration: {end_time-start_time}, lits length {len(random_lits)}, wordlength: {len(random_lits[0])}, max cnf: {max_cnf_res}"])

failed_test = np.array(failed_test)
result_test = np.array(result_test)
print(f"Test Result: \n{result_test}")
print(f"Total failed tests: {len(failed_test)}")
print(f"failed tests description: \n{failed_test}")
print(f"Total tests done: {total_test*3}")
print(f"Duration: {end_time-start_time}")





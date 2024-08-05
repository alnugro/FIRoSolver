import random
from pb2cnf import PB2CNF
from pysat.solvers import Solver

def interpret_2s_complement(value, bits):
    """Interpret a binary value as a 2's complement number."""
    msb_mask = 1 << (bits - 1)
    if value & msb_mask:
        value -= 1 << bits
    return value

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
            if rounded_result <= bound+2**-fracW:
                failed_test.append([f"Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}"])
                print("\n*****************atleast failed*****************\n")
        elif case == 'atmost':
            if rounded_result >= bound-2**-fracW:
                failed_test.append([f"Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}"])
                print("\n*****************atmost failed*****************\n")
        elif case == 'equals':
            if rounded_result-(8*2**-fracW) > bound or rounded_result+(8*2**-fracW) < bound:
                failed_test.append([f"Calculated: {rounded_result} and Original: {bound}, case: {case} with fracW: {fracW}"])
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

num_lists_start = 15
num_lists_end = 20
list_length_start = 10
list_length_end = 15


# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)

    


total_test = 30  # Example number of tests, adjust as needed
failed_test = []

for i in range(total_test):
    random_lits = generate_randomized_lits(num_lists_start, num_lists_end, list_length_start, list_length_end)
    fracW_min = int(0.6*len(random_lits[0]))
    fracW_max = len(random_lits[0]) - int(0.2*len(random_lits[0]))
    # fracW = random.randint(fracW_min, fracW_max)
    fracW = 0
    bound_min = -2**(len(random_lits[0])-fracW-1)
    bound_max = 2**(len(random_lits[0])-fracW-1)
    bound = random.randint(bound_min, bound_max)
    top_var = max(max(lit_group) for lit_group in random_lits)
    weight_min = -2**(len(random_lits[0])-fracW-1)
    weight_max = 2**(len(random_lits[0])-fracW-1)

    random_weights = [random.randint(weight_min, weight_max) for _ in range(len(random_lits))]

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

print(f"Total failed tests: {len(failed_test)}")
print(f"failed tests description: {failed_test}")
print(f"Total tests done: {total_test*3}")




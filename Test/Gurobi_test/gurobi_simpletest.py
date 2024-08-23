import gurobipy as gp
from gurobipy import GRB

try:
    # Create a new model
    model = gp.Model("sum_constraints_example")

    # Create variables
    num_vars = 5
    vars = [model.addVar(name=f"x{i}") for i in range(num_vars)]

    # Set the objective (optional, not necessary if we're only interested in the constraints)
    model.setObjective(0, GRB.MINIMIZE)  # Just a dummy objective for now

    # Add constraints such that the sum of all variables equals 1
    model.addConstr(gp.quicksum(vars) == 1, "sum_constraint")

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        for v in model.getVars():
            print(f"{v.varName}: {v.x}")
        print(f"Objective value: {model.objVal}")
    else:
        print("No optimal solution found.")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")

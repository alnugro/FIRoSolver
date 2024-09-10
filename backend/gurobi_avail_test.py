
def gurobi_test():
    try:
        from gurobipy import Model, GRB
        
        # Create a new model
        model = Model("test_model")

        # Create variables
        x = model.addVar(name="x")
        y = model.addVar(name="y")

        # Set objective: maximize 3x + 4y
        model.setObjective(3 * x + 4 * y, GRB.MAXIMIZE)

        # Add constraint: x + 2y <= 14
        model.addConstr(x + 2 * y <= 14, "c1")

        # Add constraint: 3x - y >= 0
        model.addConstr(3 * x - y >= 0, "c2")

        # Add constraint: x - y <= 2
        model.addConstr(x - y <= 2, "c3")

        # Optimize the model
        model.optimize()

        # Check if an optimal solution was found
        if model.status == GRB.OPTIMAL:
            print("Optimization was successful.")
            print(f"x = {x.X}")
            print(f"y = {y.X}")
            print(f"Objective value = {model.ObjVal}")
        else:
            print(f"Optimization was unsuccessful. Status code: {model.status}")

    except Exception as e:
        print(f"Gurobi encountered an error: {e}")

# Run the test
gurobi_test()
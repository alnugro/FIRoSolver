from pysat.formula import CNF, IDPool
from pysat.pb import PBEnc
from pysat.solvers import Solver

# Function to measure time

class PBENCTest():
    def __init__(self, top_var = None):
        self.top_var=top_var
    
    def pb2cnf(self, weight,lits,bound,case):
        new_lits = []
        new_weight = []
        for i, we in enumerate(weight):
            print(lits[i])
            new_lits.extend(lits[i])
            for j in range(len(lits[i])):
                bool_weight = 2**(j)
                if j == len(lits[i])-1:
                    bool_weight = -2**(j)
                pb_weight = bool_weight * we
                new_weight.append(pb_weight)
        print (f"weight: {new_weight}")
        print (f"lits: {new_lits}")
        solver = Solver(name='cadical195')
        solver.activate_atmost()  # Ensure atmost is activated
        if case == 'atleast':
            cnf = PBEnc.atleast(lits=new_lits, weights=new_weight, bound=bound,top_id=self.top_var)
        elif case == 'atmost':
            cnf = PBEnc.atmost(lits=new_lits, weights=new_weight, bound=bound,top_id=self.top_var)
        elif case == 'equals':
            cnf = PBEnc.equals(lits=new_lits, weights=new_weight, bound=bound,top_id=self.top_var)
        else:
            raise ValueError("Unknown case type")
        
        return cnf.clauses
    
    def solve_with_native(lits, weights, bound):
        solver = Solver(name='cadical195')
        solver.activate_atmost()  # Ensure atmost is activated
        solver.add_atmost(lits, bound, weights)
        is_sat = solver.solve()
        model = solver.get_model() if is_sat else None
        solver.delete()
        return is_sat, model    

                
    # Function to solve using PBEnc
    def solve_with_pbenc(lits, weights, bound):
        cnf = PBEnc.atmost(lits=lits, weights=weights, bound=bound)
        cnf.extend(PBEnc.atleast(lits=lits, weights=weights, bound=bound).clauses)

        solver = Solver(name='cadical195')
        for clause in cnf.clauses:
            solver.add_clause(clause)
        is_sat = solver.solve()
        model = solver.get_model() if is_sat else None
        solver.delete()
        return is_sat, model

if __name__ == "__main__":
    pb = PBENCTest()
    Lits= [[1,2,3,4,5],[6,7,8,9,10]]
    Weights= [2,3]
    bound = 0
    top_var = 10
    cnf = pb.pb2cnf(Lits,Weights,bound,"atmost")
    print(cnf)


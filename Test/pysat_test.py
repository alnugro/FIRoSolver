from pysat.pb import *

cnf = PBEnc.atmost(lits=[1, 2, 3], weights=[1, 2, 3], bound=3)
[[4], [-1, -5], [-2, -5], [5, -3, -6], [6]]
cnf = PBEnc.equals(lits=[1, 2, 3], weights=[1, 2, 3], bound=3, encoding=EncType.bdd)
print(cnf.clauses)

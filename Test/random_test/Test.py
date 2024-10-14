# Bound ci,0 to be an odd number
for i in range(1, self.adder_count + 1):
    model.addConstr(c[i][0] == 1)

# Last c or c[N+1] is connected to ground, so all zeroes
for w in range(self.adder_wordlength):
    model.addConstr(c[self.adder_count + 1][w] == 0)

# Input multiplexer constraints
for i in range(1, self.adder_count + 1):
    alpha_sum = gp.LinExpr()
    beta_sum = gp.LinExpr()
    for a in range(i):
        for word in range(self.adder_wordlength):
            # Equivalent to clause1_1 and clause1_2
            model.addConstr(-alpha[i-1][a] - c[a][word] + l[i-1][word] >= -1)
            model.addConstr(-alpha[i-1][a] + c[a][word] - l[i-1][word] >= -1)

            # Equivalent to clause2_1 and clause2_2
            model.addConstr(-beta[i-1][a] - c[a][word] + r[i-1][word] >= -1)
            model.addConstr(-beta[i-1][a] + c[a][word] - r[i-1][word] >= -1)

        alpha_sum += alpha[i-1][a]
        beta_sum += beta[i-1][a]

    # AtMost and AtLeast constraints for alpha and beta sums
    model.addConstr(alpha_sum == 1)
    model.addConstr(beta_sum == 1)

# Left Shifter constraints
gamma = [[model.addVar(vtype=GRB.BINARY, name=f'gamma_{i}_{k}') for k in range(self.adder_wordlength - 1)] for i in range(1, self.adder_count + 1)]
s = [[model.addVar(vtype=GRB.BINARY, name=f's_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]

for i in range(1, self.adder_count + 1):
    gamma_sum = gp.LinExpr()
    for k in range(self.adder_wordlength - 1):
        for j in range(self.adder_wordlength - 1 - k):
            # Equivalent to clause3_1 and clause3_2
            model.addConstr(-gamma[i-1][k] - l[i-1][j] + s[i-1][j+k] >= -1)
            model.addConstr(-gamma[i-1][k] + l[i-1][j] - s[i-1][j+k] >= -1)

        gamma_sum += gamma[i-1][k]

    model.addConstr(gamma_sum == 1)

    for kf in range(1, self.adder_wordlength - 1):
        for b in range(kf):
            # Equivalent to clause4, clause5, and clause6
            model.addConstr(-gamma[i-1][kf] - s[i-1][b] >= -1)
            model.addConstr(-gamma[i-1][kf] - l[i-1][self.adder_wordlength - 1] + l[i-1][self.adder_wordlength - 2 - b] >= -1)
            model.addConstr(-gamma[i-1][kf] + l[i-1][self.adder_wordlength - 1] - l[i-1][self.adder_wordlength - 2 - b] >= -1)

    # Equivalent to clause7_1 and clause7_2
    model.addConstr(-l[i-1][self.adder_wordlength - 1] + s[i-1][self.adder_wordlength - 1] >= 0)
    model.addConstr(l[i-1][self.adder_wordlength - 1] - s[i-1][self.adder_wordlength - 1] >= 0)

# Delta selector constraints
delta = [model.addVar(vtype=GRB.BINARY, name=f'delta_{i}') for i in range(1, self.adder_count + 1)]
u = [[model.addVar(vtype=GRB.BINARY, name=f'u_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]
x = [[model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]

for i in range(1, self.adder_count + 1):
    for word in range(self.adder_wordlength):
        # Equivalent to clause8_1 and clause8_2
        model.addConstr(-delta[i-1] - s[i-1][word] + x[i-1][word] >= -1)
        model.addConstr(-delta[i-1] + s[i-1][word] - x[i-1][word] >= -1)

        # Equivalent to clause9_1 and clause9_2
        model.addConstr(-delta[i-1] - r[i-1][word] + u[i-1][word] >= -1)
        model.addConstr(-delta[i-1] + r[i-1][word] - u[i-1][word] >= -1)

        # Equivalent to clause10_1 and clause10_2
        model.addConstr(delta[i-1] - s[i-1][word] + u[i-1][word] >= 0)
        model.addConstr(delta[i-1] + s[i-1][word] - u[i-1][word] >= 0)

        # Equivalent to clause11_1 and clause11_2
        model.addConstr(delta[i-1] - r[i-1][word] + x[i-1][word] >= 0)
        model.addConstr(delta[i-1] + r[i-1][word] - x[i-1][word] >= 0)

# XOR constraints
epsilon = [model.addVar(vtype=GRB.BINARY, name=f'epsilon_{i}') for i in range(1, self.adder_count + 1)]
y = [[model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]

for i in range(1, self.adder_count + 1):
    for word in range(self.adder_wordlength):
        # Equivalent to clause12, clause13, clause14, clause15
        model.addConstr(u[i-1][word] + epsilon[i-1] - y[i-1][word] >= 0)
        model.addConstr(u[i-1][word] - epsilon[i-1] + y[i-1][word] >= 0)
        model.addConstr(-u[i-1][word] + epsilon[i-1] + y[i-1][word] >= 0)
        model.addConstr(-u[i-1][word] - epsilon[i-1] - y[i-1][word] >= -2)

# Ripple carry constraints
z = [[model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]
cout = [[model.addVar(vtype=GRB.BINARY, name=f'cout_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]

for i in range(1, self.adder_count + 1):
    # Clauses for sum = a ⊕ b ⊕ cin at 0
    model.addConstr(x[i-1][0] + y[i-1][0] + epsilon[i-1] - z[i-1][0] >= 0)
    model.addConstr(x[i-1][0] + y[i-1][0] - epsilon[i-1] + z[i-1][0] >= 0)
    model.addConstr(x[i-1][0] - y[i-1][0] + epsilon[i-1] + z[i-1][0] >= 0)
    model.addConstr(-x[i-1][0] + y[i-1][0] + epsilon[i-1] + z[i-1][0] >= 0)
    model.addConstr(-x[i-1][0] - y[i-1][0] - epsilon[i-1] + z[i-1][0] >= -2)
    model.addConstr(-x[i-1][0] - y[i-1][0] + epsilon[i-1] - z[i-1][0] >= -2)
    model.addConstr(-x[i-1][0] + y[i-1][0] - epsilon[i-1] - z[i-1][0] >= -2)
    model.addConstr(x[i-1][0] - y[i-1][0] - epsilon[i-1] - z[i-1][0] >= -2)

    # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b))
    model.addConstr(-x[i-1][0] - y[i-1][0] + cout[i-1][0] >= -1)
    model.addConstr(x[i-1][0] + y[i-1][0] - cout[i-1][0] >= 0)
    model.addConstr(-x[i-1][0] - epsilon[i-1] + cout[i-1][0] >= -1)
    model.addConstr(x[i-1][0] + epsilon[i-1] - cout[i-1][0] >= 0)
    model.addConstr(-y[i-1][0] - epsilon[i-1] + cout[i-1][0] >= -1)
    model.addConstr(y[i-1][0] + epsilon[i-1] - cout[i-1][0] >= 0)

    for kf in range(1, self.adder_wordlength):
        # Clauses for sum = a ⊕ b ⊕ cin at kf
        model.addConstr(x[i-1][kf] + y[i-1][kf] + cout[i-1][kf-1] - z[i-1][kf] >= 0)
        model.addConstr(x[i-1][kf] + y[i-1][kf] - cout[i-1][kf-1] + z[i-1][kf] >= 0)
        model.addConstr(x[i-1][kf] - y[i-1][kf] + cout[i-1][kf-1] + z[i-1][kf] >= 0)
        model.addConstr(-x[i-1][kf] + y[i-1][kf] + cout[i-1][kf-1] + z[i-1][kf] >= 0)
        model.addConstr(-x[i-1][kf] - y[i-1][kf] - cout[i-1][kf-1] + z[i-1][kf] >= -2)
        model.addConstr(-x[i-1][kf] - y[i-1][kf] + cout[i-1][kf-1] - z[i-1][kf] >= -2)
        model.addConstr(-x[i-1][kf] + y[i-1][kf] - cout[i-1][kf-1] - z[i-1][kf] >= -2)
        model.addConstr(x[i-1][kf] - y[i-1][kf] - cout[i-1][kf-1] - z[i-1][kf] >= -2)

        # Clauses for cout = (a AND b) OR (cin AND (a ⊕ b)) at kf
        model.addConstr(-x[i-1][kf] - y[i-1][kf] + cout[i-1][kf] >= -1)
        model.addConstr(x[i-1][kf] + y[i-1][kf] - cout[i-1][kf] >= 0)
        model.addConstr(-x[i-1][kf] - cout[i-1][kf-1] + cout[i-1][kf] >= -1)
        model.addConstr(x[i-1][kf] + cout[i-1][kf-1] - cout[i-1][kf] >= 0)
        model.addConstr(-y[i-1][kf] - cout[i-1][kf-1] + cout[i-1][kf] >= -1)
        model.addConstr(y[i-1][kf] + cout[i-1][kf-1] - cout[i-1][kf] >= 0)

    # Adjusted constraint for the last bit
    model.addConstr(epsilon[i-1] + x[i-1][self.adder_wordlength-1] + u[i-1][self.adder_wordlength-1] - z[i-1][self.adder_wordlength-1] >= 0)
    model.addConstr(epsilon[i-1] - x[i-1][self.adder_wordlength-1] - u[i-1][self.adder_wordlength-1] + z[i-1][self.adder_wordlength-1] >= -1)
    model.addConstr(-epsilon[i-1] + x[i-1][self.adder_wordlength-1] - u[i-1][self.adder_wordlength-1] - z[i-1][self.adder_wordlength-1] >= -2)
    model.addConstr(-epsilon[i-1] - x[i-1][self.adder_wordlength-1] + u[i-1][self.adder_wordlength-1] + z[i-1][self.adder_wordlength-1] >= -1)

# Right shift constraints
zeta = [[model.addVar(vtype=GRB.BINARY, name=f'zeta_{i}_{k}') for k in range(self.adder_wordlength - 1)] for i in range(1, self.adder_count + 1)]

for i in range(1, self.adder_count + 1):
    zeta_sum = gp.LinExpr()
    for k in range(self.adder_wordlength - 1):
        for j in range(self.adder_wordlength - 1 - k):
            # Equivalent to clause48_1 and clause48_2
            model.addConstr(-zeta[i-1][k] - z[i-1][j+k] + c[i][j] >= -1)
            model.addConstr(-zeta[i-1][k] + z[i-1][j+k] - c[i][j] >= -1)

        zeta_sum += zeta[i-1][k]

    model.addConstr(zeta_sum == 1)

    for kf in range(1, self.adder_wordlength - 1):
        for b in range(kf):
            # Equivalent to clause49_1, clause49_2, clause50
            model.addConstr(-zeta[i-1][kf] - z[i-1][self.adder_wordlength - 1] + c[i][self.adder_wordlength - 2 - b] >= -1)
            model.addConstr(-zeta[i-1][kf] + z[i-1][self.adder_wordlength - 1] - c[i][self.adder_wordlength - 2 - b] >= -1)
            model.addConstr(-zeta[i-1][kf] - z[i-1][b] >= -1)

    # Equivalent to clause51_1 and clause51_2
    model.addConstr(-z[i-1][self.adder_wordlength - 1] + c[i][self.adder_wordlength - 1] >= 0)
    model.addConstr(z[i-1][self.adder_wordlength - 1] - c[i][self.adder_wordlength - 1] >= 0)

# Set connected coefficient
connected_coefficient = half_order + 1 - self.avail_dsp

# Solver connection
theta = [[model.addVar(vtype=GRB.BINARY, name=f'theta_{i}_{m}') for m in range(half_order + 1)] for i in range(self.adder_count + 2)]
iota = [model.addVar(vtype=GRB.BINARY, name=f'iota_{m}') for m in range(half_order + 1)]
t = [[model.addVar(vtype=GRB.BINARY, name=f't_{m}_{w}') for w in range(self.adder_wordlength)] for m in range(half_order + 1)]

iota_sum = gp.LinExpr()
for m in range(half_order + 1):
    theta_or = gp.LinExpr()
    for i in range(self.adder_count + 2):
        for word in range(self.adder_wordlength):
            # Equivalent to clause52_1 and clause52_2
            model.addConstr(-theta[i][m] - iota[m] - c[i][word] + t[m][word] >= -2)
            model.addConstr(-theta[i][m] - iota[m] + c[i][word] - t[m][word] >= -2)
        theta_or += theta[i][m]
    model.addConstr(theta_or >= 1)

for m in range(half_order + 1):
    iota_sum += iota[m]

model.addConstr(iota_sum == connected_coefficient)

# Left Shifter in result module
# k is the shift selector
o = [[model.addVar(vtype=GRB.BINARY) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
phi = [[model.addVar(vtype=GRB.BINARY) for k in range(self.adder_wordlength - 1)] for m in range(half_order + 1)]

for m in range(half_order + 1):
    phi_sum = gp.LinExpr()
    for k in range(self.adder_wordlength - 1):
        for j in range(self.adder_wordlength - 1 - k):
            model.addConstr(-phi[m][k] - t[m][j] + o[m][j + k] >= -1)
            model.addConstr(-phi[m][k] + t[m][j] - o[m][j + k] >= -1)
        phi_sum += phi[m][k]
    # AtMost and AtLeast (phi_sum == 1)
    model.addConstr(phi_sum == 1)
    for kf in range(1, self.adder_wordlength - 1):
        for b in range(kf):
            model.addConstr(-phi[m][kf] - o[m][b] >= -1)
            model.addConstr(-phi[m][kf] - t[m][self.adder_wordlength - 1] + t[m][self.adder_wordlength - 2 - b] >= -1)
            model.addConstr(-phi[m][kf] + t[m][self.adder_wordlength - 1] - t[m][self.adder_wordlength - 2 - b] >= -1)

    model.addConstr(-t[m][self.adder_wordlength - 1] + o[m][self.adder_wordlength - 1] >= 0)
    model.addConstr(t[m][self.adder_wordlength - 1] - o[m][self.adder_wordlength - 1] >= 0)

rho = [model.addVar(vtype=GRB.BINARY) for m in range(half_order + 1)]
o_xor = [[model.addVar(vtype=GRB.BINARY) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
h_ext = [[model.addVar(vtype=GRB.BINARY) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]
cout_res = [[model.addVar(vtype=GRB.BINARY) for w in range(self.adder_wordlength)] for m in range(half_order + 1)]

# XOR constraints
for m in range(half_order + 1):
    for word in range(self.adder_wordlength):
        model.addConstr(o[m][word] + rho[m] - o_xor[m][word] >= 0)
        model.addConstr(o[m][word] - rho[m] + o_xor[m][word] >= 0)
        model.addConstr(-o[m][word] + rho[m] + o_xor[m][word] >= 0)
        model.addConstr(-o[m][word] - rho[m] - o_xor[m][word] >= -2)

# Ripple carry constraints
for m in range(half_order + 1):
    model.addConstr(o_xor[m][0] + rho[m] - h_ext[m][0] >= 0)
    model.addConstr(o_xor[m][0] - rho[m] + h_ext[m][0] >= 0)
    model.addConstr(-o_xor[m][0] + rho[m] + h_ext[m][0] >= 0)
    model.addConstr(-o_xor[m][0] - rho[m] - h_ext[m][0] >= -2)

    model.addConstr(o_xor[m][0] - cout_res[m][0] >= 0)
    model.addConstr(-o_xor[m][0] - rho[m] + cout_res[m][0] >= -1)
    model.addConstr(o_xor[m][0] + rho[m] - cout_res[m][0] >= 0)
    model.addConstr(rho[m] - cout_res[m][0] >= 0)

    for word in range(1, self.adder_wordlength):
        model.addConstr(o_xor[m][word] + cout_res[m][word - 1] - h_ext[m][word] >= 0)
        model.addConstr(o_xor[m][word] - cout_res[m][word - 1] + h_ext[m][word] >= 0)
        model.addConstr(-o_xor[m][word] + cout_res[m][word - 1] + h_ext[m][word] >= 0)
        model.addConstr(-o_xor[m][word] - cout_res[m][word - 1] - h_ext[m][word] >= -2)

        model.addConstr(o_xor[m][word] - cout_res[m][word] >= 0)
        model.addConstr(-o_xor[m][word] - cout_res[m][word - 1] + cout_res[m][word] >= -1)
        model.addConstr(o_xor[m][word] + cout_res[m][word - 1] - cout_res[m][word] >= 0)
        model.addConstr(cout_res[m][word - 1] - cout_res[m][word] >= 0)

# Solver connection
for m in range(half_order + 1):
    for word in range(self.adder_wordlength):
        if word <= self.wordlength - 1:
            # Equivalent to clause58 and clause59
            model.addConstr(-h[m][word] + h_ext[m][word] >= 0)
            model.addConstr(h[m][word] - h_ext[m][word] >= 0)
        else:
            model.addConstr(-h[m][self.wordlength - 1] + h_ext[m][word] >= 0)
            model.addConstr(h[m][self.wordlength - 1] - h_ext[m][word] >= 0)

if self.adder_depth > 0:
    # Binary variables for psi_alpha and psi_beta
    psi_alpha = [[model.addVar(vtype=GRB.BINARY, name=f'psi_alpha_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.adder_count+1)]
    psi_beta = [[model.addVar(vtype=GRB.BINARY, name=f'psi_beta_{i}_{d}') for d in range(self.adder_depth)] for i in range(1, self.adder_count+1)]

    for i in range(1, self.adder_count+1):
        psi_alpha_sum = []
        psi_beta_sum = []
        # Adjusted constraints for psi_alpha and psi_beta
        model.addConstr(-psi_alpha[i-1][0] + alpha[i-1][0] >= 0)
        model.addConstr(-psi_beta[i-1][0] + beta[i-1][0] >= 0)

        psi_alpha_sum.append(psi_alpha[i-1][0])
        psi_beta_sum.append(psi_beta[i-1][0])

        if self.adder_depth == 1:
            continue

        for d in range(1, self.adder_depth):
            for a in range(i-1):
                # Adjusted constraints for psi_alpha and psi_beta
                model.addConstr(-psi_alpha[i-1][d] + alpha[i-1][a] >= 0)
                model.addConstr(-psi_alpha[i-1][d] + psi_alpha[a][d-1] >= 0)
                model.addConstr(-psi_beta[i-1][d] + beta[i-1][a] >= 0)
                model.addConstr(-psi_beta[i-1][d] + psi_beta[a][d-1] >= 0)

            psi_alpha_sum.append(psi_alpha[i-1][d])
            psi_beta_sum.append(psi_beta[i-1][d])

        # AtMost and AtLeast for psi_alpha_sum and psi_beta_sum
        model.addConstr(sum(psi_alpha_sum) == 1)
        model.addConstr(sum(psi_beta_sum) == 1)

if solver_option == 'try_h_zero_count' or solver_option == 'try_max_h_zero_count':
    model.setObjective(0, GRB.MINIMIZE)
    if h_zero_count == None:
        raise TypeError("Gurobi: h_zero_count in Barebone cant be empty when try_h_zero_count is chosen")

    h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
    h_zero_sum = 0
    for m in range(half_order + 1):
        for w in range(self.wordlength):
            model.addGenConstrIndicator(h_zero[m], True, h[m][w] == 0)
        h_zero_sum += h_zero[m]
    model.addConstr(h_zero_sum >= h_zero_count)

elif solver_option == 'find_max_zero':
    h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
    h_zero_sum = 0
    for m in range(half_order + 1):
        for w in range(self.wordlength):
            model.addGenConstrIndicator(h_zero[m], True, h[m][w] == 0)
        h_zero_sum += h_zero[m]
    model.setObjective(h_zero_sum, GRB.MAXIMIZE)

else:
    model.setObjective(0, GRB.MAXIMIZE)

print("solver running")
start_time = time.time()
model.optimize()

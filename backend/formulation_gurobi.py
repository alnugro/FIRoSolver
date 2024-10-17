import numpy as np
import math
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB

try:
    from .solver_func import SolverFunc
except: 
    from solver_func import SolverFunc



class FIRFilterGurobi:
    def __init__(self, 
                 filter_type, 
                 half_order, 
                 freqx_axis, 
                 upperbound_lin, 
                 lowerbound_lin, 
                 ignore_lowerbound, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 coef_accuracy,
                 intW,
                 run_auto_thread=False,
                 intfeasttol=None,
                 ):
        
        
        
        self.filter_type = filter_type
        self.half_order = half_order
        self.freqx_axis = freqx_axis

        self.h_res = []
        self.gain_res = 0

        self.wordlength = wordlength
        self.adder_count = adder_count

        self.upperbound_lin=upperbound_lin
        self.lowerbound_lin=lowerbound_lin

        self.coef_accuracy = coef_accuracy
        self.intW = intW
        self.fracW = self.wordlength - self.intW

        self.gain_upperbound= gain_upperbound
        self.gain_lowerbound= gain_lowerbound

        self.ignore_lowerbound = ignore_lowerbound

        self.adder_depth = adder_depth
        self.avail_dsp = avail_dsp
        self.adder_wordlength = self.wordlength + adder_wordlength_ext
        self.result_model = {}
        self.run_auto_thread = False





    def get_solver_func_dict(self):
        input_data_sf = {
        'filter_type': self.filter_type,
        }

        return input_data_sf
    
    def run_barebone(self, thread, minmax_option = None, h_zero_count = None):
        

        self.h_res = []
        self.gain_res = 0
        target_result = {}
        half_order = self.half_order - 1 #-1 is because i am lazy to change the code
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

        # print(f"upperbound_lin: {self.upperbound_lin}")
        # print(f"lowerbound_lin: {self.lowerbound_lin}")


         # linearize the bounds
        internal_upperbound_lin = [round((f)*(2**(self.fracW)), self.coef_accuracy) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [round((f)*(2**(self.fracW)), self.coef_accuracy) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = round(self.ignore_lowerbound*(2**self.fracW), self.coef_accuracy)


        # print("Running Gurobi with the following parameters:")
        # print(f"thread: {thread}")
        # print(f"minmax_option: {minmax_option}")
        # print(f"h_zero_count: {h_zero_count}")
        # print(f"filter_type: {self.filter_type}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")


        model = gp.Model()
        if self.run_auto_thread == False:
            model.setParam('Threads', thread)
            
        model.setParam('OutputFlag', 0)


        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")



        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= round(cm*(-1*(2**w)), self.coef_accuracy)
                    else: cm_word_prod= round(cm*((2**w)), self.coef_accuracy)
                    h_sum_temp += h[m][w]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain * internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain * -internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain * internal_lowerbound_lin[omega])

        if minmax_option == 'try_h_zero_count':
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

        elif minmax_option == 'find_max_zero':
            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                for w in range(self.wordlength):
                    model.addGenConstrIndicator(h_zero[m], True, h[m][w] == 0)
                h_zero_sum += h_zero[m]
            model.setObjective(h_zero_sum, GRB.MAXIMIZE)
            
        else:
            model.setObjective(0, GRB.MINIMIZE)



            
        
        print("Gurobi: Barebone running")
        model.optimize()

        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'

            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = True if h[m][w].X > 0.5 else False
                    # print(f"h{m}{w} = {bool_value}")
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w==self.wordlength-1:
                        fir_coef += -2**(w-self.fracW)  * (1 if bool_value else 0)
                    elif w < self.fracW:                   
                        fir_coef += 2**(-1*(self.fracW-w)) * (1 if bool_value else 0)
                    else:
                        fir_coef += 2**(w-self.fracW) * (1 if bool_value else 0)
                
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res=gain.x
            #print("gain Coeffs: ", self.gain_res)
            if minmax_option == 'try_h_zero_count':
                target_result.update({
                        'satisfiability' : satisfiability,
                        'h_res' : self.h_res,
                        'gain_res' : self.gain_res,
                    })
            elif minmax_option == 'find_max_zero':
                #asign h_zero value
                h_zero_sum_res= 0
                for m in range(half_order + 1):
                    h_zero_sum_res += h_zero[m].X
                target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : self.h_res,
                    'max_h_zero' : h_zero_sum_res,
                    'gain_res' : self.gain_res,
                    
                })

            else:
                target_result.update({
                        'satisfiability' : satisfiability,
                        'h_res' : self.h_res,
                        'gain_res' : self.gain_res,
                    })
                
            

                      
        else:
            print("Gurobi: Unsatisfiable")
            target_result.update({
                    'satisfiability' : satisfiability,
                })
            
        model.terminate()  # Dispose of the model
        del model

        print(target_result)

        return target_result
    
    def run_barebone_real(self,thread, minmax_option, h_zero_count = None, h_target = None):
        self.h_res = []
        self.gain_res = []
        target_result = {}
        half_order = self.half_order - 1 #-1 is because i am lazy to change the code
        print("Gurobi solver called")
        sf = SolverFunc(self.get_solver_func_dict())

        # print(f"upperbound_lin: {self.upperbound_lin}")
        # print(f"lowerbound_lin: {self.lowerbound_lin}")

         # linearize the bounds
        internal_upperbound_lin = [math.floor((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [math.ceil((f)*(10**self.coef_accuracy)) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = self.ignore_lowerbound*(10**self.coef_accuracy)

        # print("Running Gurobi with the following parameters:")
        # print(f"thread: {thread}")
        # print(f"minmax_option: {minmax_option}")
        # print(f"h_zero_count: {h_zero_count}")
        # print(f"filter_type: {self.filter_type}")
        # print(f"freqx_axis: {self.freqx_axis}")
        # print(f"upperbound_lin: {internal_upperbound_lin}")
        # print(f"lowerbound_lin: {internal_lowerbound_lin}")
        # print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        # print(f"gain_upperbound: {self.gain_upperbound}")
        # print(f"gain_lowerbound: {self.gain_lowerbound}")
        # print(f"wordlength: {self.wordlength}")
        # print(f"fracW: {self.fracW}")
        

        
        model = gp.Model(f"presolve_model_{minmax_option}")
        if self.run_auto_thread == False:
            model.setParam('Threads', thread)
        model.setParam('OutputFlag', 0)

        h_upperbound = ((2**(self.intW-1))-1)+(1-2**-self.fracW)
        h_lowerbound = -2**(self.intW-1)

        h = [model.addVar(vtype=GRB.CONTINUOUS,lb=h_lowerbound, ub=h_upperbound, name=f'h_{a}') for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")

        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                cm_word_prod= int(cm*(10** self.coef_accuracy))
                h_sum_temp += h[m]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain*internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain*-internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain*internal_lowerbound_lin[omega])
        
        print(f"Gurobi barebone_real: {minmax_option}")
                
        if minmax_option == 'find_max_zero':
            print(minmax_option)
            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.setObjective(h_zero_sum, GRB.MAXIMIZE)

        elif minmax_option == 'find_min_gain':
            if h_zero_count == None:
                model.setObjective(gain, GRB.MINIMIZE)
            else:
                h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
                h_zero_sum = 0
                for m in range(half_order + 1):
                    model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                    h_zero_sum += h_zero[m]
                model.addConstr(h_zero_sum >= h_zero_count)
                
                model.setObjective(gain, GRB.MINIMIZE)


        elif minmax_option == 'maximize_h' or minmax_option == 'minimize_h':
            if h_target == None:
                raise TypeError("Gurobi barebone_real: h_target cant be empty when maximize_h/minimize_h is chosen")
            
            if h_zero_count == None:
                raise TypeError("Gurobi barebone_real: h_zero_count cant be empty when find_min_gain is chosen")

            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.addConstr(h_zero_sum >= h_zero_count)

            if minmax_option == 'maximize_h':
                model.setObjective(h[h_target], GRB.MAXIMIZE)

            elif minmax_option == 'minimize_h':
                model.setObjective(h[h_target], GRB.MINIMIZE)
        
        elif minmax_option == 'maximize_h_without_zero' or minmax_option == 'minimize_h_without_zero':
            if h_target == None:
                raise TypeError("Gurobi barebone_real: h_target cant be empty when maximize_h_without_zero/minimize_h_without_zero is chosen")

            if minmax_option == 'maximize_h_without_zero':
                model.setObjective(h[h_target], GRB.MAXIMIZE)

            elif minmax_option == 'minimize_h_without_zero':
                model.setObjective(h[h_target], GRB.MINIMIZE)
        
        elif minmax_option == 'try_h_zero_count':
            model.setObjective(0, GRB.MINIMIZE)
            if h_zero_count == None:
                raise TypeError("Gurobi: h_zero_count in Barebone cant be empty when try_h_zero_count is chosen")

            h_zero = [model.addVar(vtype=GRB.BINARY, name=f'h_zero_{a}') for a in range(half_order + 1)]
            h_zero_sum = 0
            for m in range(half_order + 1):
                model.addGenConstrIndicator(h_zero[m], True, h[m] == 0)
                h_zero_sum += h_zero[m]
            model.addConstr(h_zero_sum >= h_zero_count)

        else:
            model.setObjective(0, GRB.MINIMIZE)


        print("Gurobi: MinMax running")
        model.optimize()
        satisfiability = 'unsat'

        if model.status == GRB.OPTIMAL:
            satisfiability = 'sat'

            for m in range(half_order + 1):
                h_value = h[m].X
                self.h_res.append(h_value)
            # print("FIR Coeffs calculated: ",self.h_res)

            self.gain_res = gain.x
            # print("gain Coeffs: ", self.gain_res)

            if minmax_option == 'find_max_zero':
                #asign h_zero value
                h_zero_sum_res= 0
                for m in range(half_order + 1):
                    h_zero_sum_res += h_zero[m].X
                target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : self.h_res,
                    'max_h_zero' : h_zero_sum_res,
                    'gain_res' : self.gain_res,

                })

            elif minmax_option == 'find_min_gain':
                target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : self.h_res,
                    'min_gain' : self.gain_res,
                })
            
            elif minmax_option == 'maximize_h' or minmax_option == 'minimize_h':
                target_result.update({
                    'satisfiability' : satisfiability,
                    'target_h_res' : h[h_target].X,
                    'gain_res' : self.gain_res,
                })
            elif minmax_option == 'maximize_h_without_zero' or minmax_option == 'minimize_h_without_zero':
                 target_result.update({
                    'satisfiability' : satisfiability,
                    'target_h_res' : h[h_target].X,
                })
            else:
                target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : self.h_res,
                    'gain_res' : self.gain_res
                })
                      
        else:
            print("Gurobi: Unsatisfiable")
            target_result.update({
                    'satisfiability' : satisfiability,
                    'h_res' : None,
                    'gain_res' : None
                })

        model.dispose()  # Dispose of the model
        del model

        print(target_result)

        return target_result



    def runsolver(self, thread ,presolve_result ,solver_option = None, adderm = None , h_zero_count = None):
        
        self.result_model = {}
        self.h_res = []
        self.gain_res = 0

        if adderm != None:
            self.adder_count = adderm

        half_order = self.half_order - 1 #-1 is because i am lazy to change the code
        print("solver called")
        sf = SolverFunc(self.get_solver_func_dict())

        
         # linearize the bounds
        internal_upperbound_lin = [round((f)*(2**(self.fracW)), self.coef_accuracy) if not np.isnan(f) else np.nan for f in self.upperbound_lin]
        internal_lowerbound_lin = [round((f)*(2**(self.fracW)), self.coef_accuracy) if not np.isnan(f) else np.nan for f in self.lowerbound_lin]
        internal_ignore_lowerbound = round(self.ignore_lowerbound*(2**self.fracW), self.coef_accuracy)
        # print("ignore lower than:", internal_ignore_lowerbound)

        
        model = gp.Model("fir_model")
        if self.run_auto_thread == False:
            model.setParam('Threads', thread)

        model.setParam('Presolve', 2)
        model.setParam('Method', 2)

        if solver_option == 'try_max_h_zero_count' or solver_option == 'try_h_zero_count':
            model.setParam('SolutionLimit', 1)
            model.setParam('MipFocus', 1)
            model.setParam('Cuts', 0)

        

        if adderm > 2:
            if h_zero_count == None:
                norelax = (adderm**2) *  (half_order)
            else:
                norelax = (adderm**2) *  (half_order - h_zero_count)

            model.setParam('NoRelHeurWork', norelax)
        # model.setParam('OutputFlag', 0)
        # model.setParam('TimeLimit', 1)     #timeout


        if solver_option == 'try_max_h_zero_count':
            self.gain_lowerbound = presolve_result['min_gain']
            hmax = presolve_result['hmax']
            hmin = presolve_result['hmin']
        else:
            self.gain_lowerbound = presolve_result['min_gain_without_zero']
            hmax = presolve_result['hmax_without_zero']
            hmin = presolve_result['hmin_without_zero']
        
        print("Running Gurobi with the following parameters:")
        print(f"thread: {thread}")
        print(f"adder_depth: {self.adder_depth}")
        print(f"h_zero_count: {h_zero_count}")
        print(f"filter_type: {self.filter_type}")
        print(f"freqx_axis: {self.freqx_axis}")
        print(f"upperbound_lin: {internal_upperbound_lin}")
        print(f"lowerbound_lin: {internal_lowerbound_lin}")
        print(f"ignore_lowerbound: {internal_ignore_lowerbound}")
        print(f"gain_upperbound: {self.gain_upperbound}")
        print(f"gain_lowerbound: {self.gain_lowerbound}")
        print(f"wordlength: {self.wordlength}")
        print(f"fracW: {self.fracW}")

        
        h = [[model.addVar(vtype=GRB.BINARY, name=f'h_{a}_{w}') for w in range(self.wordlength)] for a in range(half_order + 1)]
        gain = model.addVar(vtype=GRB.CONTINUOUS, lb=self.gain_lowerbound, ub=self.gain_upperbound, name="gain")



        for omega in range(len(self.freqx_axis)):
            if np.isnan(internal_lowerbound_lin[omega]):
                continue

            h_sum_temp = 0

            for m in range(half_order+1):
                cm = sf.cm_handler(m, self.freqx_axis[omega])
                for w in range(self.wordlength):
                    if w==self.wordlength-1:
                        cm_word_prod= round(cm*(-1*(2**w)), self.coef_accuracy)
                    else: cm_word_prod= round(cm*((2**w)), self.coef_accuracy)
                    h_sum_temp += h[m][w]*cm_word_prod

            model.update()
            # print(f"sum temp is{h_sum_temp}")
            model.addConstr(h_sum_temp <= gain * internal_upperbound_lin[omega])
            
            
            if internal_lowerbound_lin[omega] < internal_ignore_lowerbound:
                model.addConstr(h_sum_temp >= gain * -internal_upperbound_lin[omega])
            else:
                model.addConstr(h_sum_temp >= gain * internal_lowerbound_lin[omega])

            for m in range(half_order + 1):
                h_sum_temp = 0
                for w in range(self.wordlength):
                    weight = 0
                    if w==self.wordlength-1:
                        weight = -2**(w-self.fracW)  
                    elif w < self.fracW:                   
                        weight = 2**(-1*(self.fracW-w)) 
                    else:
                        weight = 2**(w-self.fracW)

                    h_sum_temp+= h[m][w]*weight

                model.addConstr(h_sum_temp <= hmax[m])
                model.addConstr(h_sum_temp >= hmin[m])



        # Bitshift SAT starts here

        # Define binary variables for c, l, r, alpha, beta, gamma, delta, etc.
        c = [[model.addVar(vtype=GRB.BINARY, name=f'c_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(self.adder_count + 2)]
        l = [[model.addVar(vtype=GRB.BINARY, name=f'l_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]
        r = [[model.addVar(vtype=GRB.BINARY, name=f'r_{i}_{w}') for w in range(self.adder_wordlength)] for i in range(1, self.adder_count + 1)]

        alpha = [[model.addVar(vtype=GRB.BINARY, name=f'alpha_{i}_{a}') for a in range(i)] for i in range(1, self.adder_count + 1)]
        beta = [[model.addVar(vtype=GRB.BINARY, name=f'beta_{i}_{a}') for a in range(i)] for i in range(1, self.adder_count + 1)]

        # # c0,w is always 0 except at index fracW
        # for w in range(self.fracW + 1, self.adder_wordlength):
        #     model.addConstr(c[0][w] == 0)

        # for w in range(self.fracW):
        #     model.addConstr(c[0][w] == 0)

        # model.addConstr(c[0][self.fracW] == 1)

        
        # c0,w is always 0 except at 0 so input
        for w in range(1, self.adder_wordlength):
            model.addConstr(c[0][w] == 0)

        model.addConstr(c[0][0] == 1)




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





        # print(filter_coeffs)
        # print(filter_literals)

        satisfiability = 'unsat'

       
        if model.status == GRB.OPTIMAL:
            for i in range(1, self.adder_count+1):
                for d in range(self.adder_depth):
                    psi_alpha_val = psi_alpha[i-1][d].X
                    psi_beta_val = psi_beta[i-1][d].X
                    print(f"psi_alpha_{i}_{d}: {psi_alpha_val}")
                    print(f"psi_beta_{i}_{d}: {psi_beta_val}")

            end_time = time.time()

            satisfiability = 'sat'
            print("solver sat")

            # Calculate h coefficients
            self.h_res = []
            for m in range(half_order + 1):
                fir_coef = 0
                for w in range(self.wordlength):
                    # Evaluate the boolean value from the model
                    bool_value = 1 if h[m][w].X > 0.5 else 0  # Rounding the value to 0 or 1
                    # Convert boolean to integer (0 or 1) and calculate the term
                    if w == self.wordlength - 1:
                        fir_coef += -2 ** (w - self.fracW) * bool_value
                    elif w < self.fracW:
                        fir_coef += 2 ** (-1 * (self.fracW - w)) * bool_value
                    else:
                        fir_coef += 2 ** (w - self.fracW) * bool_value
                self.h_res.append(fir_coef)
            print("FIR Coeffs calculated Final: ", self.h_res)

            h_zero_count = 0
            if solver_option:
                for m in range(half_order + 1):
                    if h_zero[m].X > 0.5:  # Ensure binary rounding
                        h_zero_count += 1

            # Store h coefficients
            self.result_model.update({"h_res": self.h_res})

            # Calculate and store gain coefficient
            self.gain_res = gain.X  # Gain is likely continuous, so no rounding here
            print("gain Coeffs: ", self.gain_res)
            self.result_model.update({"gain": self.gain_res})

            # Store h
            h_values = []
            for i in range(len(h)):
                h_row = []
                for a in range(len(h[i])):
                    value = 1 if h[i][a].X > 0.5 else 0  # Rounding binary variable
                    h_row.append(value)
                h_values.append(h_row)
            self.result_model.update({"h": h_values})

            # Store alpha selectors
            alpha_values = []
            for i in range(len(alpha)):
                alpha_row = []
                for a in range(len(alpha[i])):
                    value = 1 if alpha[i][a].X > 0.5 else 0  # Rounding binary variable
                    alpha_row.append(value)
                alpha_values.append(alpha_row)
            self.result_model.update({"alpha": alpha_values})

            # Store beta selectors
            beta_values = []
            for i in range(len(beta)):
                beta_row = []
                for a in range(len(beta[i])):
                    value = 1 if beta[i][a].X > 0.5 else 0  # Rounding binary variable
                    beta_row.append(value)
                beta_values.append(beta_row)
            self.result_model.update({"beta": beta_values})

            # Store gamma (left shift selectors)
            gamma_values = []
            for i in range(len(gamma)):
                gamma_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if gamma[i][k].X > 0.5 else 0  # Rounding binary variable
                    gamma_row.append(value)
                gamma_values.append(gamma_row)
            self.result_model.update({"gamma": gamma_values})

            # Store delta selectors
            delta_values = []
            for i in range(len(delta)):
                value = 1 if delta[i].X > 0.5 else 0  # Rounding binary variable
                delta_values.append(value)
            self.result_model.update({"delta": delta_values})

            # Store epsilon selectors
            epsilon_values = []
            for i in range(len(epsilon)):
                value = 1 if epsilon[i].X > 0.5 else 0  # Rounding binary variable
                epsilon_values.append(value)
            self.result_model.update({"epsilon": epsilon_values})

            # Store zeta (right shift selectors)
            zeta_values = []
            for i in range(len(zeta)):
                zeta_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if zeta[i][k].X > 0.5 else 0  # Rounding binary variable
                    zeta_row.append(value)
                zeta_values.append(zeta_row)
            self.result_model.update({"zeta": zeta_values})

            # Store phi selectors
            phi_values = []
            for i in range(len(phi)):
                phi_row = []
                for k in range(self.adder_wordlength - 1):
                    value = 1 if phi[i][k].X > 0.5 else 0  # Rounding binary variable
                    phi_row.append(value)
                phi_values.append(phi_row)
            self.result_model.update({"phi": phi_values})

            # Store theta array
            theta_values = []
            for i in range(len(theta)):
                theta_row = []
                for m in range(half_order + 1):
                    value = 1 if theta[i][m].X > 0.5 else 0  # Rounding binary variable
                    theta_row.append(value)
                theta_values.append(theta_row)
            self.result_model.update({"theta": theta_values})

            # Store iota array
            iota_values = []
            for m in range(len(iota)):
                value = 1 if iota[m].X > 0.5 else 0  # Rounding binary variable
                iota_values.append(value)
            self.result_model.update({"iota": iota_values})

            #store rho array
            rho_values = []
            for m in range(len(rho)):
                value = 1 if rho[m].X > 0.5 else 0
                rho_values.append(value)
            self.result_model.update({"rho": rho_values})


        elif model.Status == GRB.TIME_LIMIT:
            print("Optimization stopped due to time limit.")
            satisfiability = 'timeout'
        else:
            print("Gurobi: Unsatisfiable")
            end_time = time.time()

        print("solver stopped")
        duration = end_time - start_time
        print(f"Duration: {duration} seconds")

        # Gurobi report
        print(f"\n************Gurobi Report****************")
        print(f"Total number of variables            : {model.NumVars}")
        print(f"Total number of constraints (clauses): {model.NumConstrs}\n")

        model.dispose()  # Dispose of the model
        del model

        return self.result_model, satisfiability, h_zero_count


    

if __name__ == "__main__":

    # Test inputs
    filter_type = 0
    order_current = 20
    accuracy = 1
    wordlength = 11
    gain_upperbound = 1
    gain_lowerbound = 1
    coef_accuracy = 4
    intW = 2

    adder_count = 4
    adder_depth = 0
    avail_dsp = 0
    adder_wordlength_ext = 2
    
    
    delta = 0.071192
    passband_error = delta
    stopband_error = delta
    space = order_current * accuracy
    # Initialize freq_upper and freq_lower with NaN values
    freqx_axis = np.linspace(0, 1, space)
    freq_upper = np.full(space, np.nan)
    freq_lower = np.full(space, np.nan)

    # Manually set specific values for the elements of freq_upper and freq_lower in dB
    lower_half_point = int(0.3 * space)
    upper_half_point = int(0.5 * space)
    end_point = space

    freq_upper[0:lower_half_point] = 1 + passband_error
    freq_lower[0:lower_half_point] = 1 - passband_error

    freq_upper[upper_half_point:end_point] = 0 + stopband_error
    freq_lower[upper_half_point:end_point] = 0


    #beyond this bound lowerbound will be ignored
    ignore_lowerbound = -60

    #linearize the bound
    upperbound_lin = np.copy(freq_upper)
    lowerbound_lin = np.copy(freq_lower)
    ignore_lowerbound_lin = 10 ** (ignore_lowerbound / 20)
    
    

    # Create FIRFilter instance
    fir_filter = FIRFilterGurobi(
                 filter_type, 
                 order_current, 
                 freqx_axis, 
                 upperbound_lin, 
                 lowerbound_lin, 
                 ignore_lowerbound_lin, 
                 adder_count, 
                 wordlength, 
                 adder_depth,
                 avail_dsp,
                 adder_wordlength_ext,
                 gain_upperbound,
                 gain_lowerbound,
                 coef_accuracy,
                 intW,
                 )
            
    gurobi_thread = 15
    presolve = True
    presolve_result = {}

    if presolve:
        half_order = (order_current // 2) if filter_type == 0 or filter_type == 2 else (order_current // 2) - 1


        target_result = fir_filter.run_barebone(gurobi_thread, 'find_max_zero')
        max_h_zero = target_result['max_h_zero']

        target_result = fir_filter.run_barebone_real(gurobi_thread, 'find_min_gain',max_h_zero, None)
        min_gain = target_result['min_gain']

    
        h_max = []
        h_min = []
        for m in range(half_order + 1):
            target_result_max = fir_filter.run_barebone_real(gurobi_thread, 'maximize_h',max_h_zero, m)
            target_result_min = fir_filter.run_barebone_real(gurobi_thread, 'minimize_h',max_h_zero, m)
            h_max.append(target_result_max['target_h_res'])
            h_min.append(target_result_min['target_h_res'])
            

        presolve_result.update({
                'max_zero' : max_h_zero,
                'min_gain' : min_gain,
                'hmax' : h_max,
                'hmin' : h_min,
            })
        
        print(presolve_result)
    # Run solver
    target_result = fir_filter.runsolver(0,presolve_result,'try_max_h_zero_count',2, 6)
    print(target_result)
    # fir_filter.run_barebone(0,None,None)
    # target_result = fir_filter.run_barebone(1,'minimize_h',None, 0)
    
    # target_result = fir_filter.run_barebone_real(0,None)
    # target_result = fir_filter.run_barebone_real(1,'maximize_h', 0)



    # fir_filter.run_barebone_real(1,'find_max_zero')
    # fir_filter.run_barebone_real(1,'find_min_gain')

    
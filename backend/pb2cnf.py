import numpy as np
from pysat.solvers import Solver

try:
    from .rat2bool import Rat2bool
except:
    from rat2bool import Rat2bool



'''''
fracW is the fractional wordlength from the given literalls wordlength, example: given lits is in the length of 10 with fracW of 4, it means it has the size of 6q4, or in other word, 6 integer size and 4 fractional fixed point value.
top_var is the top most int variable that you made in your solver, the auxilary variable will start from top_var+1
'''''

class PB2CNF():
    def __init__(self, weight_wordlength ,top_var = None):
        
        self.var = 1
        #top var
        if top_var != None:
            self.var = top_var+1
        self.r2b = Rat2bool()
        self.weight_wordlength = weight_wordlength

    def addition_matcher(self, res, a, b, cin):
        cnf_addit = []
        cout = None

        # Case 1: (int, int, int)
        if isinstance(a, int) and isinstance(b, int) and isinstance(cin, int):
            a_int, b_int, cin_int = a, b, cin
            # print(f"Case1: a == {a_int}, b == {b_int}, cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([a_int, b_int, cin_int, -res])
            cnf_addit.append([a_int, b_int, -cin_int, res])
            cnf_addit.append([a_int, -b_int, cin_int, res])
            cnf_addit.append([-a_int, b_int, cin_int, res])
            cnf_addit.append([-a_int, -b_int, -cin_int, res])
            cnf_addit.append([-a_int, -b_int, cin_int, -res])
            cnf_addit.append([-a_int, b_int, -cin_int, -res])
            cnf_addit.append([a_int, -b_int, -cin_int, -res])

            cnf_addit.append([-a_int, -b_int, cout])
            cnf_addit.append([a_int, b_int, -cout])
            cnf_addit.append([-a_int, -cin_int, cout])
            cnf_addit.append([a_int, cin_int, -cout])
            cnf_addit.append([-b_int, -cin_int, cout])
            cnf_addit.append([b_int, cin_int, -cout])

            return cnf_addit, cout

        # Case 2: (int, 'zero', 'zero')
        elif isinstance(a, int) and b == 'zero' and cin == 'zero':
            a_int = a
            # print(f"Case2: a == {a_int}, b == 'zero', cin == 'zero'")
            cout = 'zero'
            cnf_addit.append([-res, a_int])
            cnf_addit.append([res, -a_int])
            return cnf_addit, cout

        # Case 3: (int, 'zero', 'one')
        elif isinstance(a, int) and b == 'zero' and cin == 'one':
            a_int = a
            # print(f"Case3: a == {a_int}, b == 'zero', cin == 'one'")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -a_int])
            cnf_addit.append([res, a_int])

            cnf_addit.append([-cout, a_int])
            cnf_addit.append([cout, -a_int])
            return cnf_addit, cout

        # Case 4: (int, 'zero', int)
        elif isinstance(a, int) and b == 'zero' and isinstance(cin, int):
            a_int, cin_int = a, cin
            # print(f"Case4: a == {a_int}, b == 'zero', cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([a_int, cin_int, -res])
            cnf_addit.append([a_int, -cin_int, res])
            cnf_addit.append([-a_int, cin_int, res])
            cnf_addit.append([-a_int, -cin_int, -res])

            cnf_addit.append([a_int, -cout])
            cnf_addit.append([-a_int, -cin_int, cout])
            cnf_addit.append([a_int, cin_int, -cout])
            cnf_addit.append([cin_int, -cout])
            return cnf_addit, cout

        # Case 5: ('zero', int, int)
        elif a == 'zero' and isinstance(b, int) and isinstance(cin, int):
            b_int, cin_int = b, cin
            # print(f"Case5: a == 'zero', b == {b_int}, cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([b_int, cin_int, -res])
            cnf_addit.append([b_int, -cin_int, res])
            cnf_addit.append([-b_int, cin_int, res])
            cnf_addit.append([-b_int, -cin_int, -res])

            cnf_addit.append([b_int, -cout])
            cnf_addit.append([cin_int, -cout])
            cnf_addit.append([-b_int, -cin_int, cout])
            cnf_addit.append([b_int, cin_int, -cout])
            return cnf_addit, cout

        # Case 6: ('zero', 'zero', 'zero')
        elif a == 'zero' and b == 'zero' and cin == 'zero':
            # print(f"Case6: a == 'zero', b == 'zero', cin == 'zero'")
            cout = 'zero'
            cnf_addit.append([-res])
            return cnf_addit, cout

        # Case 7: ('zero', 'one', 'zero')
        elif a == 'zero' and b == 'one' and cin == 'zero':
            # print(f"Case7: a == 'zero', b == 'one', cin == 'zero'")
            cout = 'zero'
            cnf_addit.append([res])
            return cnf_addit, cout

        # Case 8: ('one', 'zero', 'zero')
        elif a == 'one' and b == 'zero' and cin == 'zero':
            # print(f"Case8: a == 'one', b == 'zero', cin == 'zero'")
            cout = 'zero'
            cnf_addit.append([res])
            return cnf_addit, cout

        # Case 9: ('one', 'one', 'zero')
        elif a == 'one' and b == 'one' and cin == 'zero':
            # print(f"Case9: a == 'one', b == 'one', cin == 'zero'")
            cout = 'one'
            cnf_addit.append([-res])
            return cnf_addit, cout

        # Case 10: ('zero', 'zero', 'one')
        elif a == 'zero' and b == 'zero' and cin == 'one':
            # print(f"Case10: a == 'zero', b == 'zero', cin == 'one'")
            cout = 'zero'
            cnf_addit.append([res])
            return cnf_addit, cout

        # Case 11: ('zero', 'one', 'one')
        elif a == 'zero' and b == 'one' and cin == 'one':
            # print(f"Case11: a == 'zero', b == 'one', cin == 'one'")
            cout = 'one'
            cnf_addit.append([-res])
            return cnf_addit, cout

        # Case 12: ('one', 'zero', 'one')
        elif a == 'one' and b == 'zero' and cin == 'one':
            # print(f"Case12: a == 'one', b == 'zero', cin == 'one'")
            cout = 'one'
            cnf_addit.append([-res])
            return cnf_addit, cout

        # Case 13: ('one', 'one', 'one')
        elif a == 'one' and b == 'one' and cin == 'one':
            # print(f"Case13: a == 'one', b == 'one', cin == 'one'")
            cout = 'one'
            cnf_addit.append([res])
            return cnf_addit, cout

        # Case 14: (int, 'one', 'zero')
        elif isinstance(a, int) and b == 'one' and cin == 'zero':
            a_int = a
            # print(f"Case14: a == {a_int}, b == 'one', cin == 'zero'")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -a_int])
            cnf_addit.append([res, a_int])

            cnf_addit.append([-cout, a_int])
            cnf_addit.append([cout, -a_int])
            return cnf_addit, cout

        # Case 15: ('zero', int, 'zero')
        elif a == 'zero' and isinstance(b, int) and cin == 'zero':
            b_int = b
            # print(f"Case15: a == 'zero', b == {b_int}, cin == 'zero'")
            cout = 'zero'
            cnf_addit.append([-res, b_int])
            cnf_addit.append([res, -b_int])
            return cnf_addit, cout

        # Case 16: ('one', int, 'zero')
        elif a == 'one' and isinstance(b, int) and cin == 'zero':
            b_int = b
            # print(f"Case16: a == 'one', b == {b_int}, cin == 'zero'")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -b_int])
            cnf_addit.append([res, b_int])

            cnf_addit.append([-cout, b_int])
            cnf_addit.append([cout, -b_int])
            return cnf_addit, cout

        # Case 17: (int, int, 'zero')
        elif isinstance(a, int) and isinstance(b, int) and cin == 'zero':
            a_int, b_int = a, b
            # print(f"Case17: a == {a_int}, b == {b_int}, cin == 'zero'")
            cout = self.aux_var_setter()
            cnf_addit.append([a_int, b_int, -res])
            cnf_addit.append([a_int, -b_int, res])
            cnf_addit.append([-a_int, b_int, res])
            cnf_addit.append([-a_int, -b_int, -res])

            cnf_addit.append([-a_int, -b_int, cout])
            cnf_addit.append([a_int, b_int, -cout])
            cnf_addit.append([a_int, -cout])
            cnf_addit.append([b_int, -cout])
            return cnf_addit, cout

        # Case 18: (int, 'one', 'one')
        elif isinstance(a, int) and b == 'one' and cin == 'one':
            a_int = a
            # print(f"Case18: a == {a_int}, b == 'one', cin == 'one'")
            cout = 'one'
            cnf_addit.append([-res, a_int])
            cnf_addit.append([res, -a_int])
            return cnf_addit, cout

        # Case 19: ('zero', int, 'one')
        elif a == 'zero' and isinstance(b, int) and cin == 'one':
            b_int = b
            # print(f"Case19: a == 'zero', b == {b_int}, cin == 'one'")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -b_int])
            cnf_addit.append([res, b_int])

            cnf_addit.append([-cout, b_int])
            cnf_addit.append([cout, -b_int])
            return cnf_addit, cout

        # Case 20: ('one', int, 'one')
        elif a == 'one' and isinstance(b, int) and cin == 'one':
            b_int = b
            # print(f"Case20: a == 'one', b == {b_int}, cin == 'one'")
            cout = 'one'
            cnf_addit.append([-res, b_int])
            cnf_addit.append([res, -b_int])
            return cnf_addit, cout

        # Case 21: (int, int, 'one')
        elif isinstance(a, int) and isinstance(b, int) and cin == 'one':
            a_int, b_int = a, b
            # print(f"Case21: a == {a_int}, b == {b_int}, cin == 'one'")
            cout = self.aux_var_setter()
            cnf_addit.append([a_int, b_int, res])
            cnf_addit.append([-a_int, -b_int, res])
            cnf_addit.append([-a_int, b_int, -res])
            cnf_addit.append([a_int, -b_int, -res])

            cnf_addit.append([-a_int, -b_int, cout])
            cnf_addit.append([a_int, b_int, -cout])
            cnf_addit.append([-a_int, cout])
            cnf_addit.append([-b_int, cout])
            return cnf_addit, cout

        # Case 22: ('zero', 'zero', int)
        elif a == 'zero' and b == 'zero' and isinstance(cin, int):
            cin_int = cin
            # print(f"Case22: a == 'zero', b == 'zero', cin == {cin_int}")
            cout = 'zero'
            cnf_addit.append([-res, cin_int])
            cnf_addit.append([res, -cin_int])
            return cnf_addit, cout

        # Case 23: ('zero', 'one', int)
        elif a == 'zero' and b == 'one' and isinstance(cin, int):
            cin_int = cin
            # print(f"Case23: a == 'zero', b == 'one', cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -cin_int])
            cnf_addit.append([res, cin_int])

            cnf_addit.append([-cout, cin_int])
            cnf_addit.append([cout, -cin_int])
            return cnf_addit, cout

        # Case 24: ('one', 'zero', int)
        elif a == 'one' and b == 'zero' and isinstance(cin, int):
            cin_int = cin
            # print(f"Case24: a == 'one', b == 'zero', cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([-res, -cin_int])
            cnf_addit.append([res, cin_int])

            cnf_addit.append([-cout, cin_int])
            cnf_addit.append([cout, -cin_int])
            return cnf_addit, cout

        # Case 25: ('one', 'one', int)
        elif a == 'one' and b == 'one' and isinstance(cin, int):
            cin_int = cin
            # print(f"Case25: a == 'one', b == 'one', cin == {cin_int}")
            cout = 'one'
            cnf_addit.append([-res, -cin_int])
            cnf_addit.append([res, cin_int])
            return cnf_addit, cout

        # Case 26: (int, 'one', int)
        elif isinstance(a, int) and b == 'one' and isinstance(cin, int):
            a_int, cin_int = a, cin
            # print(f"Case26: a == {a_int}, b == 'one', cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([a_int, cin_int, res])
            cnf_addit.append([-a_int, -cin_int, res])
            cnf_addit.append([-a_int, cin_int, -res])
            cnf_addit.append([a_int, -cin_int, -res])

            cnf_addit.append([-a_int, cout])
            cnf_addit.append([-a_int, -cin_int, cout])
            cnf_addit.append([a_int, cin_int, -cout])
            cnf_addit.append([-cin_int, cout])
            return cnf_addit, cout

        # Case 27: ('one', int, int)
        elif a == 'one' and isinstance(b, int) and isinstance(cin, int):
            b_int, cin_int = b, cin
            # print(f"Case27: a == 'one', b == {b_int}, cin == {cin_int}")
            cout = self.aux_var_setter()
            cnf_addit.append([b_int, cin_int, res])
            cnf_addit.append([-b_int, -cin_int, res])
            cnf_addit.append([-b_int, cin_int, -res])
            cnf_addit.append([b_int, -cin_int, -res])

            cnf_addit.append([-b_int, cout])
            cnf_addit.append([-cin_int, cout])
            cnf_addit.append([-b_int, -cin_int, cout])
            cnf_addit.append([b_int, cin_int, -cout])
            return cnf_addit, cout

        # No case found
        else:
            # print("no case found")
            raise InterruptedError("caser error, no case found. contact the developer")


    def atmost(self, weight, lits, bounds, fracW):
        self.input_validation(lits,weight)
        self.weight_bounds_validation(weight,len(lits[0]), bounds,fracW)
        cnf = self.run_pb2cnf(weight, lits, bounds, fracW,"atmost")
        return cnf

    def atleast(self, weight, lits, bounds, fracW):
        self.input_validation(lits,weight)
        self.weight_bounds_validation(weight,len(lits[0]),bounds,fracW)
        cnf = self.run_pb2cnf(weight, lits, bounds, fracW, "atleast")
        return cnf
    
    def equal(self, weight, lits, bounds, fracW):
        self.input_validation(lits,weight)
        self.weight_bounds_validation(weight,len(lits[0]),bounds,fracW)
        cnf = self.run_pb2cnf(weight, lits, bounds, fracW, "equal")
        return cnf
    
    def equal_card_one(self, lits):
        cnf = []
        cnf.append(lits)
        cnf_temp = []
        for i in range(len(lits)):
            for j in range(i+1,len(lits)):
                cnf_temp.append([-lits[i],-lits[j]])
        cnf += cnf_temp
        return cnf
    
    def equal_card(self,lits,bound):
        ext_lits = []
        weight = []
        for lit in lits:
            ext_lits.append([lit,'zero'])
            weight.append(1)
        cnf = self.run_pb2cnf(weight, ext_lits, bound, 0, "equal")
        return cnf
    

    def remove_zeroes_weight(self, we, li):
        weight = we[:]  # Create a copy of the original weight list
        lits = li[:]    # Create a copy of the original lits list
        deleted_cnf = []
        # Iterate in reverse to avoid index issues while deleting elements
        for i in range(len(weight) - 1, -1, -1):
            
            if all(x == 0 for x in weight[i]):
                deleted_cnf += ([[-l ]for l in lits[i]])
                del weight[i]
                del lits[i]
                #todo assert the deleted lits to 0
                print(f"Lits and Weight at pos {i} are deleted, because it will go to 0 with given fracW")
        return weight, lits , deleted_cnf


        
    
    def run_pb2cnf(self, weight, literalls, bounds, fracW , case):
        wordlength = len(literalls[0])
        weight_csd = self.r2b.frac2csd(weight, self.weight_wordlength, fracW).tolist()
        weight_csd ,lits , deleted_lits_cnf= self.remove_zeroes_weight(weight_csd,literalls)


        cnf_final = []
        cnf_final += deleted_lits_cnf
        sum_wordlength = 2*len(lits[0])
        lits_counts = len(lits)
        bounds *= -1
    
        bounds_list = [bounds]
        bounds_bool = self.r2b.frac2bool2s(bounds_list, wordlength, fracW).tolist()

        


        adder_model, cnf_list_generator = self.adder_model_list_generator(weight_csd, lits)
        cnf_final += cnf_list_generator

        for i in range(lits_counts):
            #add 1 to lsb if it was inversed and not added by zero. only the case if it were not bitshifted and multiplied by minus 1 or csd[0] = -1, so it is always on the adder_model[i][1] position

            while len(adder_model[i]) > 3:
                # print("deleting one line in adder graph")

                sum_sub_res = []
                for j in range(sum_wordlength):
                    sum_sub_res.append(self.aux_var_setter())

                # #print("sum sub res: ",sum_sub_res) 
                cnf_final+=(self.list_addition(sum_sub_res, adder_model[i][1], adder_model[i][2], False))

                #assign the aux model to the 2nd pos then delete the 3rd
                adder_model[i][1]= sum_sub_res
                del adder_model[i][2]

                # #print(f"cnf final is: {cnf_final}\n\n")

                #continue until you only have 3 variables
                
            # print("Model After sum ",adder_model)

            # print("deleting adder graph done")
            if len(adder_model[i]) == 2:
               
               cnf_final += (self.list_equalization(adder_model[i][0],adder_model[i][1]))
               # print(f"cnf final is: {cnf_final}")

               del adder_model[i][1]
               # print("updated :",adder_model)

            elif len(adder_model[i]) == 3:
                
                


                #sum 2nd and 3rd pos in list
                cnf_final+=(self.list_addition(adder_model[i][0], adder_model[i][1], adder_model[i][2], False))
              


        

                del adder_model[i][2]
                # print("First ",adder_model[i][0])

                # print("second ",adder_model[i][1])
                del adder_model[i][1]
                # print("third ",adder_model)

            else:
                raise RuntimeError(f"length of addermodel {i} = {len(adder_model[i])} is not valid: this should be impossible some random charged particle from space might have striked your pc")
            
        max_integer_sum_adder_csd = (2**sum_wordlength)*lits_counts
        extended_wordlength = int(np.ceil(np.log2(max_integer_sum_adder_csd)))
        #print(extended_wordlength)

        sum_adder_model= self.generate_sum_adder_model(adder_model, extended_wordlength)


        if fracW == 0:
            # print(f"\nbefore: {sum_adder_model}")
            bounds2 = bounds*2**fracW
            bounds_list2 = [bounds2]
            bounds_bool2 = self.r2b.frac2bool2s(bounds_list2, extended_wordlength, fracW).tolist()
            bounds_bool_str = self.r2b.bool2str(bounds_bool2[0])

        else:
            bounds_list = [bounds]
            bounds_bool = self.r2b.frac2bool2s(bounds_list, wordlength, fracW).tolist()
            for i in range(wordlength, extended_wordlength):
                #extend signbit for bounds bool
                bounds_bool[0].append(bounds_bool[0][-1])
            #all we do is left shifts thats why the fractional part has to go so multiply bounds by 2**fracW
            bounds_bool[0]  = self.r2b.right_shift(bounds_bool[0] ,fracW)
            bounds_bool_str = self.r2b.bool2str(bounds_bool[0])

        sum_adder_model.append(bounds_bool_str)
        


        # print(sum_adder_model)
        #add all the sum
        while len(sum_adder_model) > 1:
            # print("deleting the sum of adder graph")
            sum_sub_res = []
            for j in range(extended_wordlength):
                sum_sub_res.append(self.aux_var_setter())

            # print(f"Sum this {sum_adder_model[0]} with this {sum_adder_model[1]}")
            cnf_final+=(self.list_addition(sum_sub_res, sum_adder_model[0], sum_adder_model[1], False))

            #assign the aux model to the 2nd pos then delete the 3rd
            sum_adder_model[0] = sum_sub_res
            # print(f"This is deleted {sum_adder_model[1]}\n")
            del sum_adder_model[1]

            #continue until you only have 1 variables

        
        sum_adder_model_1d = [sublist for sublist in sum_adder_model[0]]
        #print(f"adder in the end: {sum_adder_model_1d}")

        cnf_final+=self.bound_case(sum_adder_model_1d,case)
        # print(f"\nCNF final: {cnf_final} ")

        # print("deleting done")

        return cnf_final

            
            
            
            
        # print("cnf final: ", cnf_final)
        # print(inversion_list)
        # print(self.var)

    
    def bound_case(self,sum_adder_model_1d,case):
        cnf_bound = []
        if case == 'atmost':
            for i in range(len(sum_adder_model_1d)-1):
                cnf_bound.append([sum_adder_model_1d[-1], -sum_adder_model_1d[i]])
        elif case == 'atleast':
            for i in range(len(sum_adder_model_1d)-1):
                cnf_bound.append([-sum_adder_model_1d[-1]])
        elif case == 'equal':
            for i in range(len(sum_adder_model_1d)):
                cnf_bound.append([-sum_adder_model_1d[i]])

        else:
            raise ValueError(f"Value {case} cant be determined")


        return self.simplify_cnf(cnf_bound)
    
    def generate_sum_adder_model(self, adder_model, ext_word):
        sum_adder_model = adder_model
        for i in range(len(adder_model)):
            for j in range(len(adder_model[i][0]), ext_word):
                sum_adder_model[i][0].append(adder_model[i][0][-1])
    
        sum_adder_model2d = [sublist[0] for sublist in sum_adder_model]
        return sum_adder_model2d



    
    def list_addition(self, lst_result, lst_a, lst_b, inversion_f):
        cnf_addit = []
        inversion_flag = inversion_f
        cout = []
        for i in range(len(lst_result)):
            cout.append(None)

        for i in range(len(lst_result)):
            if i == 0 and inversion_flag:
                cnf_temp,cout_temp = self.addition_matcher(lst_result[i], lst_a[i], lst_b[i], 'one')

            elif i == 0 and inversion_flag == False:
                cnf_temp,cout_temp = self.addition_matcher(lst_result[i],lst_a[i], lst_b[i], 'zero')
            else:
                cin = cout[i-1]  #carry cin
                cnf_temp,cout_temp = self.addition_matcher(lst_result[i],lst_a[i], lst_b[i], cin)
            
            if cout_temp !=None:
                cout[i] = cout_temp
                cnf_addit.extend(cnf_temp)
            else:
                raise ValueError("cout is somehow None: contact developer")
            
        #bound the 
        # print(f"\nCNF is this: {cnf_addit} ")
        # print(f"Cout is this: {cout} ")


        # print(f"final CNF is this: {cnf_addit} \n\n\n")
        # print(f" Simplified CNF{self.simplify_cnf(cnf_addit)}\n\n")
        return self.simplify_cnf(cnf_addit)
       

    def aux_var_setter(self):
        self.var += 1
        return self.var-1

    
    def simplify_cnf(self,cnf):
        simplified_cnf = []

        for clause in cnf:
            # Remove duplicate literals within the clause
            unique_literals = list(set(clause))

            # Check if the clause is trivially true
            is_trivially_true = False
            literal_set = set(unique_literals)
            for literal in unique_literals:
                if -literal in literal_set:
                    is_trivially_true = True
                    break

            if not is_trivially_true:
                simplified_cnf.append(unique_literals)

        return simplified_cnf

    def list_equalization(self, lst1,lst2):
        cnf_equal = []
        for i in range(len(lst1)):
            if lst1[i] == 'zero' and (lst2[i] != 'zero' or lst2[i] != 'one'):
                cnf_equal.append([-1*(lst2[i])])
                continue
            if lst2[i] == 'zero' and (lst1[i] != 'zero' or lst1[i] != 'one'):
                cnf_equal.append([-1*(lst1[i])])
                continue
            if lst1[i] == 'one' and (lst2[i] != 'zero' or lst2[i] != 'one'):
                cnf_equal.append([lst2[i]])
                continue
            if lst2[i] == 'one' and (lst1[i] != 'zero' or lst1[i] != 'one'):
                cnf_equal.append([lst1[i]])
                continue
            
            cnf_equal.append([-lst1[i], lst2[i]])
            cnf_equal.append([lst1[i], -lst2[i]])

        # print ("cnf equal: ",cnf_equal)
        return cnf_equal

    def list_invertion(self, lst, lst_res):
        cnf_inversion = []
        one_str= self.r2b.frac2str([0], len(lst), 0)
        lst_b = ['one' if l == 'zero' else 'zero' if l == 'one' else -l for l in lst]
        cnf_inversion=self.list_addition(lst_result=lst_res, lst_a=one_str[0], lst_b=lst_b, inversion_f=True)
        return cnf_inversion



    def adder_model_list_generator(self, weight_csd, lits):
        inversion_cnf = []
        sum_wordlength = 2*len(lits[0])
        wordlength = len(lits[0])
        lits_counts = len(lits)

        adder_model = [[[[] for _ in range(sum_wordlength)] for _ in range(1)] for _ in range(lits_counts)]

        inversion_list= []

        for i in range(lits_counts):
            for j in range(sum_wordlength):
                adder_model[i][0][j]=self.aux_var_setter()
            

        # print("CSD: ",weight_csd)

        #shifting csd parts
        for i in range(lits_counts):
            for j in range(len(weight_csd[0])): #csd wordlenght to be exact
                adder_submodule = ['zero' for i in range(sum_wordlength)]
                # print("i: ",i)
                # print("j: ",j)
                if weight_csd[i][j] == 0:
                    continue

                elif weight_csd[i][j] == 1:
                    for k in range(j, j+wordlength):
                        # print(f"k is {k}")
                        # print(f"k-j is {k-j}")
                        adder_submodule[k] = lits[i][k-j]

                    
                    for l in range(j+wordlength,sum_wordlength):
                        #extend the sign bit
                        adder_submodule[l] = lits[i][-1]

                    adder_model[i].append(adder_submodule)
                    # print(adder_submodule)

                elif weight_csd[i][j] == -1:
                    for k in range(j,j+wordlength):
                        # print(f"k is {k}")
                        # print(f"k-j-1 is {k-j}")
                        adder_submodule[k] = lits[i][k-j]

                    
                    for l in range(j+wordlength,sum_wordlength):
                        #extend the sign bit
                        adder_submodule[l] = lits[i][-1]

                    #invert the submodule and sum it with 1
                    sum_sub_res = []
                    for j in range(sum_wordlength):
                        sum_sub_res.append(self.aux_var_setter())

                    inversion_cnf += self.list_invertion(adder_submodule,sum_sub_res)

                    adder_submodule = sum_sub_res

                    adder_model[i].append(adder_submodule)
        
        #print(adder_model)
        return adder_model,inversion_cnf

    def input_validation(self, lits, weight):
        if not lits:
            raise ValueError("given lits is empty")
        
        first_length = len(lits[0])
        seen = set()

        for sublist in lits:
            if len(sublist) != first_length:
                raise ValueError("Litterals should have the same dimension")
            for item in sublist:
                if item == 0:
                    raise ValueError("Litterals cannot be zero, SAT solver won't support this")
                
        if len(lits) != len(weight):
            raise ValueError("literalls and weight are not in the same size!")
            
    def weight_bounds_validation(self, weight, wordlength, bounds, fracW):
        max_integer_pos_value = 2**(self.weight_wordlength-fracW-1)-1
        max_integer_neg_value = -1*max_integer_pos_value #its the same to keep the negation from getting overflown
        #print(max_integer_pos_value)
        for w in weight:
            if int(w) > max_integer_pos_value or int(w) < max_integer_neg_value:
                raise ValueError(f"given wordlength for int is too short for the weight: {w}")

        if int(bounds) > max_integer_pos_value or int(bounds) < max_integer_neg_value:
            raise ValueError(f"given wordlength for int is too short for given bounds: {bounds} \n keep them between -2**(wordlength-fracW-1)-1 and 2**(wordlength-fracW-1)-1 to avoid negation overflow")

 

        
        

if __name__ == "__main__":
    #Example Use
    # lits = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
    lits = [[1, 2, 3, 4, 5]]
    weight = [1]
    bounds = 2
    fracW = 0

    top_var = max(max(lit_group) for lit_group in lits)
    print(top_var)

    # lits = [1,2,3,4,5,6]
    # top_var = 6

    
    pb = PB2CNF(top_var=top_var, weight_wordlength=10)
    cnf = pb.equal(weight,lits, bounds,0)
    # cnf = pb.equal_card(lits,5)

    solver = Solver(name='Cadical195')

    for clause in cnf:
        solver.add_clause(clause)
    is_sat = solver.solve()

    if is_sat:
        model = solver.get_model()
        print("SAT")
        print(model)
    else: print("unsat")

    



    
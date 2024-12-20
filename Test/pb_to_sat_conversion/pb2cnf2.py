import numpy as np
from rat2bool import Rat2bool
from pysat.solvers import Solver
import math





class PB2CNF():
    '''''
    fracW is the fractional wordlength from the given literalls wordlength, example: given lits is in the length of 10 with fracW of 4, it means it has the size of 6q4, or in other word, 6 integer size and 4 fractional fixed point value.
    top_var is the top most int variable that you made in your solver, the auxilary variable will start from top_var+1
    if you have a fractional number as the weight, the solver might propagate some error depending on your fracW size
    '''''
    def __init__(self, top_var = None):
        
        self.var = 1
        #top var
        if top_var != None:
            self.var = top_var+1
        self.r2b = Rat2bool()

    
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
    
    def atmost_cardinality(self, lits, bounds):
        pass
    def atleast_cardinality(self, lits, bounds):
        pass
    def equal_cardinality(self, lits, bounds):
        pass

    def equal_card_one(self, lits):
        cnf = []
        cnf.append(lits)
        cnf_temp = []
        for i in range(len(lits)):
            for j in range(i+1,len(lits)):
                cnf_temp.append([-lits[i],-lits[j]])
                print(cnf_temp)
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
        print("deleted cnf",deleted_cnf)
        return weight, lits , deleted_cnf


        
    
    def run_pb2cnf(self, weight, literalls, bounds, fracW , case):
        wordlength = len(literalls[0])
        weight_csd = self.r2b.frac2csd(weight, wordlength, fracW).tolist()
        weight_csd ,lits , deleted_lits_cnf= self.remove_zeroes_weight(weight_csd,literalls)
        if not lits:
            raise ValueError("all The Lits was remove due to all the weight of zeroes.")

        cnf_final = []
        cnf_final += deleted_lits_cnf
        sum_wordlength = 2*len(lits[0])
        lits_counts = len(lits)
        

        #print(f"weight is : {weight}")
        #initialize Rational to boolean class
        print(f"weight csd is : {weight_csd}")
    
        
        


        adder_model, cnf_list_generator = self.adder_model_list_generator(weight_csd, lits, fracW)

        cnf_final += cnf_list_generator
        
        # print(" Original Model ",adder_model)

        #print("original, ",adder_model)
        for i in range(lits_counts):
            while len(adder_model[i]) > 1:
                # print("deleting one line in adder graph")
                # #print("sum sub res: ",sum_sub_res) 
                cnf_temp,sum_res_temp=(self.list_addition(adder_model[i][0], adder_model[i][1], False))

                #assign the aux model to the 2nd pos then delete the 3rd
                adder_model[i][0]= sum_res_temp
                cnf_final+= cnf_temp
                del adder_model[i][1]
                # #print(f"cnf final is: {cnf_final}\n\n")
                #continue until you only have 1 variables
            # print("Model After sum ",adder_model)

            # print("deleting adder graph done")

            if len(adder_model[i]) > 1:
                raise RuntimeError(f"length of addermodel {i} = {len(adder_model[i])} is not valid: this should be impossible some random charged particle from space might have striked your pc")

        # print(f"adder after deletion: {adder_model}")

        #create an extended wordlength    
        max_integer_sum_adder_csd = (2**sum_wordlength)*lits_counts
        extended_wordlength = int(math.ceil(math.log2(max_integer_sum_adder_csd)))
        #print(extended_wordlength)

        sum_adder_model= self.generate_sum_adder_model(adder_model, extended_wordlength)

        bounds *= -1
        # print(f"wl {wordlength}")
        # print(f"sum {sum_wordlength}")        
        # print(f"bounds {bounds}")
        # print(f"frac: {fracW}")
        
        #only use this bounds if fracW == 0, else it will be way too inacurate
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


        # print(f"bool1 {bounds_bool_str}\n")
        

        # print(f"Sum adder on 1: {sum_adder_model}")
        #add all the sum
        while len(sum_adder_model) > 2:
            sum_sub_res = []
            # print(f"Sum this {sum_adder_model[0]} with this {sum_adder_model[1]}")
            cnf_temp, sum_sub_res=(self.list_addition(sum_adder_model[1], sum_adder_model[2], False))
            cnf_final+= cnf_temp
            #assign the aux model to the 2nd pos then delete the 3rd
            sum_adder_model[1] = sum_sub_res
            # print(f"This is deleted {sum_adder_model[1]}\n")
            del sum_adder_model[2]

            #continue until you only have 2 variables

        #delete it one last time with variable assertion turned on
        cnf_temp, sum_sub_res=(self.list_addition(sum_adder_model[0], sum_adder_model[1], True))
        cnf_final+= cnf_temp
        #assign the aux model to the 1st pos then delete the 2nd
        sum_adder_model[0] = sum_sub_res
        # print(f"This is deleted {sum_adder_model[1]}\n")
        del sum_adder_model[1]
        # print(f"Sum adder on 2: {sum_adder_model}")

        
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



    
    def list_addition(self, lst_a, lst_b, assert_var_flag):
        cnf_addit = []
        lst_result = []
        lst_result_temp = []
        cout = []
        if len(lst_a) != len(lst_b):
            raise ValueError(f"List addition is not valid list_a: {lst_a} and lst_b: {lst_b} doesn't have a same size!")
        
        for i in range(len(lst_b)):
            cout.append(None)
            lst_result.append(None)

        for i in range(len(lst_b)):
            if i == 0:
                cnf_temp,cout_temp,lst_result_temp= self.addition_matcher(lst_a[i], lst_b[i], 'zero', assert_var_flag)

            elif i == len(lst_b)-1:
                cin = cout[i-1]  #carry cin
                cnf_temp,cout_temp,lst_result_temp = self.addition_matcher(lst_a[i], lst_b[i], cin, True)
            else:
                cin = cout[i-1]  #carry cin
                cnf_temp,cout_temp,lst_result_temp = self.addition_matcher(lst_a[i], lst_b[i], cin, assert_var_flag)
            
            if lst_result_temp!=None and cout_temp !=None:
                lst_result[i]= lst_result_temp
                cout[i] = cout_temp
                cnf_addit.extend(cnf_temp)


            else:
                raise ValueError("cout or result temp is somehow None: contact developer")
            

        # cnf_overflow = []
        # if lst_a[-1] == 'zero':
        #     sign_a = self.aux_var_setter()
        #     cnf_overflow.append([-sign_a])
        # elif lst_a[-1] == 'one':
        #     sign_a = self.aux_var_setter()
        #     cnf_overflow.append([sign_a])
        # else: sign_a = lst_a[-1]

        # if lst_b[-1] == 'zero':
        #     sign_b = self.aux_var_setter()
        #     cnf_overflow.append([-sign_b])
        # elif lst_b[-1] == 'one':
        #     sign_b = self.aux_var_setter()
        #     cnf_overflow.append([sign_b])
        # else: sign_b = lst_b[-1]

        
        # cnf_overflow.append([-sign_a, -sign_b, lst_result[-1]])
        # cnf_overflow.append([sign_a, sign_b, -lst_result[-1]])
        # cnf_addit.extend(cnf_overflow)
            
        
        #bound the 
        # print(f"\nCNF is this: {cnf_addit} ")
        # print(f"Cout is this: {cout} ")


        # print(f"final CNF is this: {cnf_addit} \n\n\n")
        # print(f" Simplified CNF{self.simplify_cnf(cnf_addit)}\n\n")
        return self.simplify_cnf(cnf_addit), lst_result
    

    
    def addition_matcher(self, a, b, cin, assert_var_flag):
        # print(f"a: {a}, b: {b}, cin: {cin}, flag: {assert_var_flag}")
        cnf_addit = []
        res = None
        cout = None
        match (a, b, cin):
            #put the most likely one on the top
            case (int() as a_int, int() as b_int, int() as cin_int):
                #print(f"Case1: a == {a_int}, b == {b_int}, cin == {cin_int}")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()
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
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case (int() as a_int, 'zero', 'zero'):
                #print(f"Case2: a == {a_int}, b == 'zero', cin == 'zero'")
                cout =  'zero'
                res = a_int

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case (int() as a_int, 'zero', 'one'):
                #print(f"Case3: a == {a_int}, b == 'zero', cin == 'one'")
                cout = a_int
                res = -a_int

                #print(cout)
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case (int() as a_int, 'zero', int() as cin_int):
                #print(f"Case4: a == {a_int}, b == 'zero', cin == {cin_int}")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()

                cnf_addit.append([a_int, cin_int, -res])
                cnf_addit.append([a_int, -cin_int, res])
                cnf_addit.append([-a_int, cin_int, res])
                cnf_addit.append([-a_int, -cin_int, -res])


                cnf_addit.append([a_int, -cout])
                cnf_addit.append([-a_int, -cin_int, cout])
                cnf_addit.append([a_int, cin_int, -cout])
                cnf_addit.append([cin_int, -cout])

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case ('zero', int() as b_int, int() as cin_int):
                #print(f"Case5: a == 'zero', b == {b_int}, cin == {cin_int}")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()

                cnf_addit.append([b_int, cin_int, -res])
                cnf_addit.append([b_int, -cin_int, res])
                cnf_addit.append([-b_int, cin_int, res])
                cnf_addit.append([-b_int, -cin_int, -res])


                cnf_addit.append([b_int, -cout])
                cnf_addit.append([cin_int, -cout])
                cnf_addit.append([-b_int, -cin_int, cout])
                cnf_addit.append([b_int, cin_int, -cout])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case ('zero', 'zero', 'zero'):
                #print(f"Case6: a == 'zero', b == 'zero', cin == 'zero'")
                cout =  'zero'
                if assert_var_flag == False:
                    res = 'zero'
                else: 
                    res = self.aux_var_setter()
                    cnf_addit.append([-res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('zero', 'one', 'zero'):
                #print(f"Case7: a == 'zero', b == 'one', cin == 'zero'")
                cout =  'zero'
                if assert_var_flag == False:
                    res= 'one'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'zero', 'zero'):
                #print(f"Case8: a == 'one', b == 'zero', cin == 'zero'")
                cout =  'zero'
                if assert_var_flag == False:
                    res= 'one'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'one', 'zero'):
                #print(f"Case9: a == 'one', b == 'one', cin == 'zero'")
                cout =  'one'
                if assert_var_flag == False:
                    res= 'zero'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([-res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case ('zero', 'zero', 'one'):
                #print(f"Case10: a == 'zero', b == 'zero', cin == 'one'")
                cout =  'zero'
                if assert_var_flag == False:
                    res= 'one'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('zero', 'one', 'one'):
                #print(f"Case11: a == 'zero', b == 'one', cin == 'one'")
                cout =  'one'
                if assert_var_flag == False:
                    res= 'zero'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([-res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'zero', 'one'):
                #print(f"Case12: a == 'one', b == 'zero', cin == 'one'")
                cout =  'one'
                if assert_var_flag == False:
                    res= 'zero'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([-res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'one', 'one'):
                #print(f"Case13: a == 'one', b == 'one', cin == 'one'")
                cout =  'one'
                if assert_var_flag == False:
                    res= 'one'
                else:
                    res = self.aux_var_setter()
                    cnf_addit.append([res])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            

            case (int() as a_int, 'one', 'zero'):
                #print(f"Case14: a == {a_int}, b == 'one', cin == 'zero'")
                cout = a_int
                res = -a_int

                cnf_addit.append([-res, -a_int])
                cnf_addit.append([res, a_int])

                cnf_addit.append([-cout, a_int])
                cnf_addit.append([cout, -a_int])

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('zero', int() as b_int, 'zero'):
                #print(f"Case15: a == 'zero', b == {b_int}, cin == 'zero'")
                cout =  'zero'
                res= b_int
                
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', int() as b_int, 'zero'):
                #print(f"Case16: a == 'one', b == {b_int}, cin == 'zero'")
                cout = b_int
                res= -b_int
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case (int() as a_int, int() as b_int, 'zero'):
                #print(f"Case17: a == {a_int}, b == {b_int}, cin == 'zero'")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()

                cnf_addit.append([a_int, b_int, -res])
                cnf_addit.append([a_int, -b_int, res])
                cnf_addit.append([-a_int, b_int, res])
                cnf_addit.append([-a_int, -b_int, -res])

                cnf_addit.append([-a_int, -b_int, cout])
                cnf_addit.append([a_int, b_int, -cout])
                cnf_addit.append([a_int, -cout])
                cnf_addit.append([b_int, -cout])
                
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
                        
            case (int() as a_int, 'one', 'one'):
                #print(f"Case18: a == {a_int}, b == 'one', cin == 'one'")
                cout =  'one'
                res= a_int
             
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('zero', int() as b_int, 'one'):
                #print(f"Case19: a == 'zero', b == {b_int}, cin == 'one'")
                cout = b_int
                res= -b_int
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', int() as b_int, 'one'):
                #print(f"Case20: a == 'one', b == {b_int}, cin == 'one'")
                cout =  'one'
                res= b_int

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case (int() as a_int, int() as b_int, 'one'):
                #print(f"Case21: a == {a_int}, b == {b_int}, cin == 'one'")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()

                cnf_addit.append([a_int, b_int, res])
                cnf_addit.append([-a_int, -b_int, res])
                cnf_addit.append([-a_int, b_int, -res])
                cnf_addit.append([a_int, -b_int, -res])

                cnf_addit.append([-a_int, -b_int, cout])
                cnf_addit.append([a_int, b_int, -cout])
                cnf_addit.append([-a_int, cout])
                cnf_addit.append([-b_int, cout])
                
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res
            
            case ('zero', 'zero', int() as cin_int):
                #print(f"Case22: a == 'zero', b == 'zero', cin == {cin_int}")
                cout =  'zero'
                res= cin_int

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('zero', 'one', int() as cin_int):
                #print(f"Case23: a == 'zero', b == 'one', cin == {cin_int}")
                cout = cin_int
                res= -cin_int
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'zero', int() as cin_int):
                #print(f"Case24: a == 'one', b == 'zero', cin == {cin_int}")
                cout = cin_int
                res= -cin_int


                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', 'one', int() as cin_int):
                #print(f"Case25: a == 'one', b == 'one', cin == {cin_int}")
                cout =  'one'
                res= -cin_int
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            

            case (int() as a_int, 'one', int() as cin_int):
                #print(f"Case26: a == {a_int}, b == 'one', cin == {cin_int}")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()
                cnf_addit.append([a_int, cin_int, res])
                cnf_addit.append([-a_int, -cin_int, res])
                cnf_addit.append([-a_int, cin_int, -res])
                cnf_addit.append([a_int, -cin_int, -res])


                cnf_addit.append([-a_int, cout])
                cnf_addit.append([-a_int, -cin_int, cout])
                cnf_addit.append([a_int, cin_int, -cout])
                cnf_addit.append([-cin_int, cout])
                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case ('one', int() as b_int, int() as cin_int):
                #print(f"Case27: a == 'one', b == {b_int}, cin == {cin_int}")
                cout = self.aux_var_setter()
                res = self.aux_var_setter()
                cnf_addit.append([b_int, cin_int, res])
                cnf_addit.append([-b_int, -cin_int, res])
                cnf_addit.append([-b_int, cin_int, -res])
                cnf_addit.append([b_int, -cin_int, -res])


                cnf_addit.append([-b_int, cout])
                cnf_addit.append([-cin_int, cout])
                cnf_addit.append([-b_int, -cin_int, cout])
                cnf_addit.append([b_int, cin_int, -cout])

                #print(f"res: {res} , cout:{cout}")
                return cnf_addit, cout, res

            case (_, _, _):
                #print("no case found")
                raise InterruptedError("caser error, no case found. contact the developer")

       

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

    def list_invertion(self, lst):
        cnf_inversion = []
        one_str= self.r2b.frac2str([1], len(lst), 0)
        lst_b = ['one' if l == 'zero' else 'zero' if l == 'one' else -l for l in lst]
        # print(f"one str{one_str}")
        cnf_inversion, lst_res=self.list_addition(lst_a=one_str[0], lst_b=lst_b, assert_var_flag=False)
        # print(f"list result {lst_res}")
        return cnf_inversion, lst_res



    def adder_model_list_generator(self, weight_csd, lits, fracW):
        inversion_cnf = []
        sum_wordlength = 2*len(lits[0])
        wordlength = len(lits[0])
        lits_counts = len(lits)

        adder_model = [[] for i in range(lits_counts)]

        #shifting csd parts
        for i in range(lits_counts):
            for j in range(wordlength): #this is csd wordlenght to be exact
                adder_submodule = ['zero' for i in range(sum_wordlength)]

                if weight_csd[i][j] == 0:
                    continue

                elif weight_csd[i][j] == 1:
                    for k in range(j,j+wordlength):
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
                    # for j in range(sum_wordlength):
                    #     sum_sub_res.append(self.aux_var_setter())

                    inversion_cnf_temp, sum_sub_res= self.list_invertion(adder_submodule)
                    inversion_cnf+= inversion_cnf_temp
                    adder_submodule = sum_sub_res

                    adder_model[i].append(adder_submodule)

        #print(f"adder model is {adder_model}")
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
        max_integer_pos_value = 2**(wordlength-fracW-1)
        max_integer_neg_value = -1*max_integer_pos_value #its the same to keep the negation from getting overflown
        #print(max_integer_pos_value)
        for w in weight:
            if int(w) > max_integer_pos_value or int(w) < max_integer_neg_value:
                raise ValueError(f"given wordlength for int is too short for the weight : {w}, please make it inside +/- 2**(wordlength-fracW-1)-1 or extend your lits")

 
    
        if int(bounds) > max_integer_pos_value or int(bounds) < max_integer_neg_value:
            raise ValueError(f"given wordlength for int is too short for given bounds: {bounds} \n keep them between -2**(wordlength-fracW-1)-1 and 2**(wordlength-fracW-1)-1 to avoid negation overflow")
 

    
        
        

if __name__ == "__main__":
    #Example Use
    # lits = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
    lits= [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    weight= [134002]
    bounds= 133246
    fracW = 4

    top_var = max(max(lit_group) for lit_group in lits)
    # print(top_var)
    # lits = [1, 2, 3, 4, 5, 6]

    pb = PB2CNF(top_var=top_var)
    cnf = pb.atleast(weight,lits,bounds,fracW)



    # cnf = pb.equal_card_one(lits)
    print("cnf: ",cnf)
    # cnf=[]
    # lits1 =['zero', 'zero', 1, 2, 3, 4, 5, 6, 7, 7, 7]
    # lits2 =['zero', 'zero', 'zero', 'zero', 1, 2, 3, 4, 5, 6, 7]
    # #result = a < 2 + a < 4 = 4a + 16a
    # res = [8,9,10,11,12,13,14,15,16,17,18]
    
    # pb = PB2CNF(top_var=21)
    # cnf += [[1],[2],[3],[4],[5]]
    # cnf += pb.list_addition(res,lits1,lits2,False )

    solver = Solver(name='Cadical195')

    for clause in cnf:
        solver.add_clause(clause)
    is_sat = solver.solve()

    if is_sat:
        model = solver.get_model()
        print("SAT")
        print(model)
    else: print("unsat")

    



    
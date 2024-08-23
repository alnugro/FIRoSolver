import numpy as np

class SolverFunc():
    def __init__(self,filter_type, order):
        self.filter_type=filter_type
        self.half_order = (order//2)
        self.overflow_count = 0

    def db_to_linear(self,db_arr):
        # Create a mask for NaN values
        nan_mask = np.isnan(db_arr)

        # Apply the conversion to non-NaN values (magnitude)
        linear_array = np.zeros_like(db_arr)
        linear_array[~nan_mask] = 10 ** (db_arr[~nan_mask] / 20)

        # Preserve NaN values
        linear_array[nan_mask] = np.nan
        return linear_array
    
    def cm_handler(self,m,omega):
        if self.filter_type == 0:
            if m == 0:
                return 1
            cm=(2*np.cos(np.pi*omega*m))
            return cm
        
        #ignore the rest, its for later use if type 1 works
        if self.filter_type == 1:
            return 2*np.cos(omega*np.pi*(m+0.5))

        if self.filter_type == 2:
            return 2*np.sin(omega*np.pi*(m-1))

        if self.filter_type == 3:
            return 2*np.sin(omega*np.pi*(m+0.5))
        
    def overflow_handler(self, input_coeffs, literal):
        max_positive_int_pbfunc = 2147483647
        max_negative_int_pbfunc = -2147483648

        self.overflow_count+=1
        overflow_coef = []
        overflow_lit = []

        if input_coeffs > max_positive_int_pbfunc:
            while input_coeffs > max_positive_int_pbfunc:
                overflow_coef.append(max_positive_int_pbfunc)
                overflow_lit.append(literal)
                input_coeffs -= max_positive_int_pbfunc
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)
            print("overflow happened in:", input_coeffs, " with literall: ", literal)
        
        elif input_coeffs < max_negative_int_pbfunc:
            while input_coeffs < max_negative_int_pbfunc:
                overflow_coef.append(max_negative_int_pbfunc)
                overflow_lit.append(literal)
                input_coeffs -= max_negative_int_pbfunc
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)
            print("overflow happened in:", input_coeffs, " with literall: ", literal)
        
        else:
            overflow_coef.append(input_coeffs)
            overflow_lit.append(literal)

        return overflow_lit, overflow_coef




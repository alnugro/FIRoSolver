import numpy as np

class Rat2bool():
    def __init__(self):
        pass

    def frac2csd(self, b_frac, nbits, nfrac):
        ntaps = len(b_frac)  # number of coefficients
        
        # Convert frac to boolean first
        A = self.abs_frac2bool(b_frac, nbits, nfrac)
        
        # Step 2: Convert binary integers to CSD
        s = np.sign(b_frac)  # signs of coefficients
        z = np.zeros((ntaps, 1), dtype=int)
        x = np.hstack((A, z))  # MSB is on right, append 0 as MSB
        Y = np.zeros((ntaps, nbits), dtype=int)
        
        for i in range(ntaps):  # coefficient index (row index)
            c = 0
            for j in range(nbits):  # binary digit index (column index)
                d = x[i, j] ^ c  # current bit xor c
                ys = x[i, j + 1] & d  # sign bit, 0 == pos, 1 == negative
                yd = (~x[i, j + 1]) & d  # data bit
                Y[i, j] = yd - ys  # signed digit
                c_next = (x[i, j] & x[i, j + 1]) | ((x[i, j] | x[i, j + 1]) & c)  # carry
                c = c_next
            
            Y[i, :] = Y[i, :] * s[i]  # multiply CSD coefficient magnitude by coefficient sign
        
        return Y

    def csd2frac(self, Y, nfrac):
        ntaps, nbits = Y.shape
        weights = np.array([2**(j - nfrac) for j in range(nbits)])
        calculated_frac = np.zeros(ntaps)
        for i in range(ntaps):
            calculated_frac[i] = np.sum(Y[i] * weights)
        
        return calculated_frac

    def abs_frac2bool(self, b_frac, nbits, nfrac):
        ntaps = len(b_frac)  # number of coefficients
        A = np.zeros((ntaps, nbits), dtype=int)
        
        # Step 1: Convert decimal integers to binary integers
        for i in range(ntaps):  # coefficient index (row index)
            integer_part = int(b_frac[i])
            fractional_part = abs(b_frac[i] - integer_part)
            u = abs(integer_part)
            for j in range(nfrac, nbits):  # binary digit index (column index)
                A[i, j] = u % 2  # coefficient magnitudes, note: MSB is on right.
                u = u // 2
            for j in range(nfrac):
                iter = nfrac-j-1
                fractional_part *= 2
                fractional_bit = int(fractional_part)
                A[i, iter] = fractional_bit
                fractional_part -= fractional_bit

        return A

    def bool2frac(self, A, nfrac):
        # It is the same as csd2frac, I just made it to make it easier to read
        calculated_frac = self.csd2frac(A, nfrac)
        return calculated_frac
    
    def bool2s2frac(self, A, nfrac):
        ntaps, nbits = A.shape
        b_frac = []
        for i in range(ntaps):
            # Reverse the binary string to convert it properly
            bin_str = ''.join(str(bit) for bit in A[i][::-1])
            int_value = int(bin_str, 2)
            if bin_str[0] == '1':  # Check for negative number in 2's complement
                int_value -= (1 << nbits)
            frac_value = int_value / (1 << nfrac)
            b_frac.append(frac_value)
        return b_frac



    def frac2bool2s(self, frac, nbits, nfrac, check_input = False):
        
        # print("i am called")
        # print(f"frac is this {frac}")
        ntaps = len(frac)  # number of coefficients
        A = np.zeros((ntaps, nbits), dtype=int)
        
        for i in range(ntaps):  # coefficient index (row index)
            if check_input:
                if int(frac[i]) > (2**(nbits-nfrac-1))-1:
                    raise ValueError(f"rat2bool: intW is too small for the given frac: {frac[i]}, please increase intW")
                if int(frac[i]) < -2**(nbits-nfrac-1):
                    raise ValueError(f"rat2bool: intW is too small for the given frac: {frac[i]}, please increase intW")

            num = frac[i]
            scale_factor = 1 << nfrac
            
            # Scale the number and round to nearest integer
            scaled_num = round(num * scale_factor)
            
            # Handle negative numbers for two's complement
            if scaled_num < 0:
                scaled_num = (1 << nbits) + scaled_num
            
            # Format the number as a binary string with leading zeros
            bin_str = f'{scaled_num:0{nbits}b}'

            # print(f"bin str bef {bin_str}")
            bin_str = bin_str.replace('-', '')
            # print(f"bin str after {bin_str}")
            
            # Ensure the binary string has the correct length
            if len(bin_str) > nbits:
                bin_str = bin_str[-nbits:]
            # elif len(bin_str) < nbits:
            #     # Pad the binary string with leading zeros if it's too short
            #     bin_str = bin_str.ljust(nbits, '0')
            

            # Reverse the binary string to match the MSB on the right convention
            A[i, :] = np.array([bit for bit in bin_str[::-1]])

        # print(f"result is this {A}")

        return A
    
    def bool2str(self, bools):
        bool_list = bools.tolist() if isinstance(bools, np.ndarray) else bools
        bool_str = []
        for i in range(len(bool_list)):
            if bool_list[i] == 0 or bool_list[i] == False:
                bool_str.append('zero')
            elif bool_list[i] == 1 or bool_list[i] == True:
                bool_str.append('one')
            else:
                raise ValueError("Input can only be 0 or 1")
        return bool_str
    
    def frac2str(self, b_frac, nbits, nfrac):
        bools = self.frac2bool2s(b_frac, nbits, nfrac)
        bool_str = []
        for i in range(len(bools)):
            bool_str.append(self.bool2str(bools[i]))
        return bool_str
    
    def frac2int(self , b_frac, nfrac):
        int_number= []
        for i in range(len(b_frac)):
            # Step 1: Multiply by 2^2 (4)
            scaled_number = b_frac[i] * (2 ** nfrac)
            
            # Step 2: Round to the nearest integer
            int_number.append(round(scaled_number))
        
        return int_number
    
    def right_shift(self, lst, n):
        """Shifts the list `lst` to the right by `n` positions."""
        if n <= 0:
            return lst  # No shifting needed if n is 0 or negative
        return [0] * n + lst[:-n]
    
    def frac2round(self, frac, nbits, nfrac, check_input = False):
        bool = self.frac2bool2s(frac,nbits,nfrac, check_input)
        frac_round = self.bool2s2frac(bool,nfrac)
        return frac_round
    
    def bool2s2frac(self,bool2s, nfrac):
        """Calculate the weighted sum based on the model using 2's complement representation."""
        frac = []
       
        for i in range(len(bool2s)):
            sum = 0
            for j in range(len(bool2s[i])):
                bool_value= bool2s[i][j]
                bool_weight = 2**(-nfrac+j)
                if j == len(bool2s[i])-1:
                    bool_weight = -2**(-nfrac+j)
                if bool_value > 0:
                    sum += bool_weight*bool_value
            frac.append(sum)
        return frac
    

                

if __name__ == "__main__":
    # Example usage:
    b = 0

    b_frac = [-11.220184543019636]  # example input coefficients
    nbits = 20  # number of bits
    nfrac = 14 # bits for fractional
    boolean = Rat2bool()
    
    # test = [-5]
    # test_res = boolean.frac2bool2s(test,5,0)

    # print ("Result ",test_res)

    
    
    # Y = boolean.frac2csd(b_frac, nbits, nfrac)
    # A = boolean.abs_frac2bool(b_frac, nbits, nfrac)
    
    # print(f"input frac:\n", b_frac)
    # print("\nleftmost is LSB")

    # print("Boolean representation (abs_frac2bool):\n", A)
    # calculated_values = boolean.bool2frac(A, nfrac)
    # print("Reconstructed values from boolean (abs_frac2bool):\n", calculated_values)

    # print("CSD representation:\n", Y)
    # calculated_values = boolean.csd2frac(Y, nfrac)
    # print("Reconstructed values from CSD:\n", calculated_values)

    twos_complement = boolean.frac2bool2s(b_frac, nbits, nfrac)
    # twos_complement= [[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]]
    print("Boolean representation (frac2bool with two's complement):\n", twos_complement)
    calculated_values_twos_complement = boolean.bool2s2frac(twos_complement, nfrac)
    print("Reconstructed values from boolean (frac2bool with two's complement):\n", calculated_values_twos_complement)

    # print(boolean.bool2str(twos_complement[0]))
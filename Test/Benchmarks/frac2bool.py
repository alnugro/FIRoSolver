import numpy as np

class Rat2bool:
    def frac2bool(self, b_frac, nbits, nfrac):
        ntaps = len(b_frac)  # number of coefficients
        A = np.zeros((ntaps, nbits), dtype=int)
        
        for i in range(ntaps):  # coefficient index (row index)
            num = b_frac[i]
            scale_factor = 1 << nfrac
            
            # Scale the number and round to nearest integer
            scaled_num = round(num * scale_factor)
            
            # Handle negative numbers for two's complement
            if scaled_num < 0:
                scaled_num = (1 << nbits) + scaled_num
            
            # Format the number as a binary string with leading zeros
            bin_str = f'{scaled_num:0{nbits}b}'
            
            # Ensure the binary string has the correct length
            if len(bin_str) > nbits:
                bin_str = bin_str[-nbits:]
            
            A[i, :] = np.array([int(bit) for bit in bin_str])
        
        return A

    def bool2frac(self, A, nfrac):
        ntaps, nbits = A.shape
        b_frac = []
        for i in range(ntaps):
            bin_str = ''.join(str(bit) for bit in A[i])
            int_value = int(bin_str, 2)
            if bin_str[0] == '1':  # Check for negative number in 2's complement
                int_value -= (1 << nbits)
            frac_value = int_value / (1 << nfrac)
            b_frac.append(frac_value)
        return b_frac

    def abs_frac2bool(self, b_frac, nbits, nfrac):
        ntaps = len(b_frac)
        A = np.zeros((ntaps, nbits), dtype=int)
        
        for i in range(ntaps):
            num = b_frac[i]
            scale_factor = 1 << nfrac
            scaled_num = int(round(num * scale_factor))
            
            if scaled_num < 0:
                scaled_num = (1 << nbits) + scaled_num
                
            bin_str = f'{scaled_num:0{nbits}b}'
            
            if len(bin_str) > nbits:
                bin_str = bin_str[-nbits:]
                
            A[i, :] = np.array([int(bit) for bit in bin_str])
        
        return A

    def frac2csd(self, b_frac, nbits, nfrac):
        # Placeholder for actual CSD conversion implementation
        pass

    def csd2frac(self, Y, nfrac):
        # Placeholder for actual CSD to fractional conversion implementation
        pass

if __name__ == "__main__":
    # Example usage:
    b_frac = [2.5, -1.75, 3, 4, 1]  # example input coefficients
    nbits = 7  # number of bits
    nfrac = 2  # bits for fractional
    boolean = Rat2bool()
    
    Y = boolean.frac2csd(b_frac, nbits, nfrac)
    A = boolean.abs_frac2bool(b_frac, nbits, nfrac)
    
    print(f"input frac:\n", b_frac)
    print("\nleftmost is LSB")

    print("Boolean representation (abs_frac2bool):\n", A)
    calculated_values = boolean.bool2frac(A, nfrac)
    print("Reconstructed values from boolean (abs_frac2bool):\n", calculated_values)

    print("CSD representation:\n", Y)
    calculated_values = boolean.csd2frac(Y, nfrac)
    print("Reconstructed values from CSD:\n", calculated_values)

    # A_twos_complement = boolean.frac2bool(b_frac, nbits, nfrac)
    # print("Boolean representation (frac2bool with two's complement):\n", A_twos_complement)
    # calculated_values_twos_complement = boolean.bool2frac(A_twos_complement, nfrac)
    # print("Reconstructed values from boolean (frac2bool with two's complement):\n", calculated_values_twos_complement)

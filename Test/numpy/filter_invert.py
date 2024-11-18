import numpy as np

def flip_filter_array(arr, filter_type, neg_flag = False):
        arr_np = np.array(arr, dtype=object)  # Specify dtype as object to allow None
        if filter_type == 0:
            # Even length: reverse the array excluding the first element (so no repeat of 0) and append the original array
            result = np.concatenate((arr_np[::-1], arr_np[1:]))
        elif filter_type == 1:
            # Odd length: reverse the array and append the original array
            result = np.concatenate((arr_np[::-1], arr_np))
            
        elif filter_type == 2:
            arr_np = np.insert(arr_np, 0, None)
            print("inv_arr:", arr_np)

            if neg_flag:
                inv_arr = arr_np[::-1]
                inv_arr = [not(i) if i is not None else i for i in inv_arr]
                result = np.concatenate((inv_arr, arr_np[1:]))
            #insert 0 at the beginning
            else:
                result = np.concatenate((arr_np[::-1], arr_np[1:]))
       
        elif filter_type == 3:
            # Odd length: reverse the array and append the original array
            if neg_flag:
                inv_arr = arr_np[::-1]
                inv_arr = [not(i) if i is not None else i for i in inv_arr]
                result = np.concatenate((inv_arr, arr_np))
            else:
                result = np.concatenate((arr_np[::-1], arr_np))
        else:
            raise ValueError("Invalid filter type")
        return result

# Example usage
arr_even = ['0', 1, 2, 3, 4, 5]
arr_even_neg = [True, True, True, True, True, True]
arr_odd = [0, 1, 2, 3, 4]
arr_odd_neg = [False for i in range(len(arr_odd))]

result_even = flip_filter_array(arr_even, 3)
result_even_neg = flip_filter_array(arr_even_neg, 3, True)
print("Even length result    :", result_even)
print("Even length neg result:", result_even_neg)


result_odd = flip_filter_array(arr_odd, 1)

print("Odd length result:", result_odd)

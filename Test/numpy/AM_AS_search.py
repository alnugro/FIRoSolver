# Given list
total_adder = [4, 2, 5, 6, 7, 1, 10]
adder_s = [6, 7, 8, 9, 10, 11, 12]



# Initialize min_value with the first element
min_value = 4

total_adder_smaller = [None for i in range(len(total_adder))]
found_min_flag = False


for i, val in enumerate(total_adder):
    print(f"value to test is {val}")    

    while True:
        if min_value-1 >= val:
            min_value = min_value-1
            found_min_flag = True
            print(f"min_value 1 is {min_value}")    

        else: break

print(min_value)

search_space = [val for val in range(1+1, 3)]

print(search_space)
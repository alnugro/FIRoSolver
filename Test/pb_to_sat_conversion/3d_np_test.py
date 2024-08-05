# Create a 3-dimensional list
three_dim_list = [
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
]

print(three_dim_list)


# Append to the first dimension
three_dim_list.append([[16, 17, 18], [19, 20, 21]])
print(three_dim_list)


# Append to the second dimension
three_dim_list[0].append([13, 14, 15])
print(three_dim_list)




# Append to the third dimension
three_dim_list[0][0].append(22)

# Print the final list
print(three_dim_list)

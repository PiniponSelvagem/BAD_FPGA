

THIS IS INCORRECT, DOES NOT WORK AS INTENDED ATM


# Define kernel and input
kernel = [
    [1, 0],
    [0, 0],
]
bias = [ 0, 0 ]

input = [
    [
        [[1, 2], [3, 4], [0, 0], [0, 0]],
        [[5, 6], [7, 8], [0, 0], [0, 0]],
    ],
    [
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ]
]

output = [
    [
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ],
    [
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]],
    ]
]

# Perform element-wise assignments using a loop
for i in range(2):
    for j in range(2):
        for k in range(2):
            #for l in range(2):
            output[i][j][k][0] = (input[i][j][k][0] * kernel[k][0]) + (input[i][j][k][1] * kernel[k][1]) + bias[0]
            output[i][j][k][1] = (input[i][j][k][0] * kernel[k][1]) + (input[i][j][k][1] * kernel[k][1]) + bias[0]

# Print the result
for row in output:
    for subrow in row:
        print(subrow)
    print()

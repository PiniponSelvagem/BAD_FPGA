file = "file.txt"

with open(file, 'r+') as file:
    # read the values from the input file
    values = file.readline().split()

    # set the file pointer to the beginning of the file
    file.seek(0)

    # write each value to a new line with a comma at the end
    for value in values:
        file.write(value + ',\n')

    # truncate the remaining content of the file
    file.truncate()
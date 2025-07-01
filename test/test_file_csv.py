from framework import file_csv

directory = '../data/healthy/' #directory with data is located in the directory prior
data = file_csv.read(directory, 100, 1800, 60) #organize the output in a SimuData structure
print(type(data))
import matplotlib.pyplot as plt
from framework import file_mat

directory = '../data/labtest_1_broken_bar/' #directory with data is located in the directory prior
data = file_mat.read(directory, 60) #organize the output in a SimuData structure
print(type(data))
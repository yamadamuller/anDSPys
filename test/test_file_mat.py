from framework import file_mat

directory = '../data/narco/struct_r1b_R1.mat' #directory with data is located in the directory prior
data = file_mat.read(directory, 4, n_periods=600, exp_num=5) #organize the output in a LaipseData structure
print(type(data))
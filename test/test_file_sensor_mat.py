from framework import file_sensor_mat

directory = '../data/benchtesting_PD/' #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, 1800, experiment_num=1, fm=60)
print(type(data))
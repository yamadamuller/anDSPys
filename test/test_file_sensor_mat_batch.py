from framework import file_sensor_mat, dsp_utils

directory = '../data/benchtesting_PD/' #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, 1800, batch=True)
print(type(data))
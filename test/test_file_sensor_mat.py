from framework import file_sensor_mat

directory = '../data/benchtesting_PD/experimento_1_carga_100__19200Hz_19200Hz.MAT' #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, 1800, fm=60)
print(type(data))
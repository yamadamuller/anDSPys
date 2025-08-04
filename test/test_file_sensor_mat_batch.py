from framework import file_sensor_mat, dsp_utils

directory = ['../data/benchtesting_PD/experimento_1_carga_100__19200Hz_19200Hz.MAT',
             '../data/benchtesting_PD/experimento_2_carga_100__19200Hz_19200Hz.MAT',
             '../data/benchtesting_PD/experimento_3_carga_100__19200Hz_19200Hz.MAT'] #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, 1800, batch=True)
print(type(data))
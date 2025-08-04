from framework import file_sensor_mat, dsp_utils
import matplotlib.pyplot as plt
import numpy as np

directory = '../data/benchtesting_PD/experimento_1_carga_100__19200Hz_19200Hz.MAT' #directory with data is located in the directory prior
plt.figure(1)
leg = []
ref_data = file_sensor_mat.read(directory, 100, 1800, fm=60, transient=True)
ref_peak = np.argmax(ref_data.i_t)
ref_peak_time = ref_data.time_grid[ref_peak]
plt.plot(ref_data.time_grid, ref_data.i_t)
leg.append('ref')
avg_data = ref_data.i_t

for i in np.arange(1,11,1):
    data = file_sensor_mat.read(directory, 100, 1800, fm=60, transient=True) #organize the output in a SimuData structure
    data_peak = np.argmax(data.i_t)
    data_peak_time = data.time_grid[data_peak]
    time_disp = ref_peak_time-data_peak_time
    placeholder_grid = data.time_grid+time_disp
    plt.plot(placeholder_grid, data.i_t)
    leg.append(f'Exp. {i}')
plt.xlim([6.66, 6.82])
plt.legend(leg)
plt.show()
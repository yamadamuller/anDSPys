from framework import file_txt, file_csv, file_sensor_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#load files and normalize by the fundamental amplitude
ni_data = file_txt.read('../data/NI/i_phase/BQ_PC_3.txt', fm=60, normalize_by=np.max)
ansys_data = file_csv.read('../data/1_broken_bar_graph', 100, 1800, fm=60, normalize_by=np.max)
qx_data = file_sensor_mat.read('../data/benchtesting_PD', 100, 1800, experiment_num=1, normalize_by=np.max)

#plots
plt.figure()
leg = []
plt.plot(ansys_data.fft_freqs, ansys_data.fft_data_dB)
leg.append('Simulation')
plt.plot(ni_data.fft_freqs, ni_data.fft_data_dB)
leg.append('Experimental')
#plt.plot(qx_data.fft_freqs, qx_data.fft_data_dB)
#leg.append('QX hardware')
plt.legend(leg, fontsize=12)
plt.xlabel('Frequency [Hz]',fontsize=12)
plt.ylabel('Magnitude of the current [dB]', fontsize=12)
plt.xlim([55,65])
plt.ylim([-80,2])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
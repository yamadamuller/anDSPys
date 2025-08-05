from framework import file_txt, file_csv, file_sensor_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#load files and normalize by the fundamental dBlitude
ni_data = file_txt.read('../data/NI/i_line/BQ_PC_IL_1.txt', fm=60, n_periods=500, normalize_by=np.max)
ansys_data = file_csv.read('../data/1_broken_bar_03082025/i_line/R', 100, 1800, fm=60, n_periods=600, normalize_by=np.max)
qx_data = file_sensor_mat.read('../data/benchtesting_PD/experimento_1_carga_100__19200Hz_19200Hz.MAT', 100, 1800, n_periods=1200, normalize_by=np.max)

#plots
plt.figure(1)
plt.subplot(3,1,1)
leg = []
lower_freq = 40
upper_freq = 80
ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
leg.append('Simulation')
ni_mask = (ni_data.fft_freqs>=lower_freq)&(ni_data.fft_freqs<=upper_freq)
plt.plot(ni_data.fft_freqs[ni_mask], ni_data.fft_data_dB[ni_mask])
leg.append(f'NI hardware')
qx_mask = (qx_data.fft_freqs >= lower_freq) & (qx_data.fft_freqs <= upper_freq)
plt.plot(qx_data.fft_freqs[qx_mask], qx_data.fft_data_dB[qx_mask])
leg.append(f'Quantum X')
plt.title(f'Line current spectra')
plt.ylabel('Amplitude [dB]')
plt.legend(leg)
plt.grid()
plt.xlim([lower_freq,upper_freq])

plt.subplot(3,1,2)
leg = []
lower_freq = 280
upper_freq = 320
ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
leg.append('Simulation')
ni_mask = (ni_data.fft_freqs>=lower_freq)&(ni_data.fft_freqs<=upper_freq)
plt.plot(ni_data.fft_freqs[ni_mask], ni_data.fft_data_dB[ni_mask])
leg.append(f'NI hardware')
qx_mask = (qx_data.fft_freqs >= lower_freq) & (qx_data.fft_freqs <= upper_freq)
plt.plot(qx_data.fft_freqs[qx_mask], qx_data.fft_data_dB[qx_mask])
leg.append(f'Quantum X')
plt.ylabel('Amplitude [dB]')
plt.legend(leg)
plt.grid()
plt.xlim([lower_freq,upper_freq])

plt.subplot(3,1,3)
leg = []
lower_freq = 400
upper_freq = 440
ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
leg.append('Simulation')
ni_mask = (ni_data.fft_freqs>=lower_freq)&(ni_data.fft_freqs<=upper_freq)
plt.plot(ni_data.fft_freqs[ni_mask], ni_data.fft_data_dB[ni_mask])
leg.append(f'NI hardware')
qx_mask = (qx_data.fft_freqs >= lower_freq) & (qx_data.fft_freqs <= upper_freq)
plt.plot(qx_data.fft_freqs[qx_mask], qx_data.fft_data_dB[qx_mask])
leg.append(f'Quantum X')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.legend(leg)
plt.grid()
plt.xlim([lower_freq,upper_freq])
plt.show()

#Batch plotting
'''
fig_counter = 1 #counter to define the figure plot
for batch_idx in range(len(ni_data.batch_data)):
    for line_idx in [0,2,7]:
        batch_obj = ni_data.batch_data[batch_idx] #load the current NIHardwareData object
        qx_obj = qx_data.batch_data[line_idx] #load the current SensorData object

        #plots
        plt.figure(fig_counter)
        plt.subplot(3,1,1)
        leg = []
        lower_freq = 40
        upper_freq = 80
        ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
        plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
        leg.append('Simulation')
        ni_mask = (batch_obj.fft_freqs>=lower_freq)&(batch_obj.fft_freqs<=upper_freq)
        plt.plot(batch_obj.fft_freqs[ni_mask], batch_obj.fft_data_dB[ni_mask])
        leg.append(f'NI hardware at exp. {batch_idx+1}')
        qx_mask = (qx_obj.fft_freqs >= lower_freq) & (qx_obj.fft_freqs <= upper_freq)
        plt.plot(qx_obj.fft_freqs[qx_mask], qx_obj.fft_data_dB[qx_mask])
        leg.append(f'QX hardware at exp. {line_idx+1}')
        plt.title(f'Line current spectra')
        plt.ylabel('Magnitude [dB]')
        plt.legend(leg)
        plt.grid()
        plt.xlim([lower_freq,upper_freq])

        plt.subplot(3,1,2)
        leg = []
        lower_freq = 280
        upper_freq = 320
        ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
        plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
        leg.append('Simulation')
        ni_mask = (batch_obj.fft_freqs>=lower_freq)&(batch_obj.fft_freqs<=upper_freq)
        plt.plot(batch_obj.fft_freqs[ni_mask], batch_obj.fft_data_dB[ni_mask])
        leg.append(f'NI hardware at exp. {batch_idx + 1}')
        qx_mask = (qx_obj.fft_freqs >= lower_freq) & (qx_obj.fft_freqs <= upper_freq)
        plt.plot(qx_obj.fft_freqs[qx_mask], qx_obj.fft_data_dB[qx_mask])
        leg.append(f'QX hardware at exp. {line_idx + 1}')
        plt.ylabel('Magnitude [dB]')
        plt.legend(leg)
        plt.grid()
        plt.xlim([lower_freq,upper_freq])

        plt.subplot(3,1,3)
        leg = []
        lower_freq = 400
        upper_freq = 440
        ansys_mask = (ansys_data.fft_freqs>=lower_freq)&(ansys_data.fft_freqs<=upper_freq)
        plt.plot(ansys_data.fft_freqs[ansys_mask], ansys_data.fft_data_dB[ansys_mask])
        leg.append('Simulation')
        ni_mask = (batch_obj.fft_freqs>=lower_freq)&(batch_obj.fft_freqs<=upper_freq)
        plt.plot(batch_obj.fft_freqs[ni_mask], batch_obj.fft_data_dB[ni_mask])
        leg.append(f'NI hardware at exp. {batch_idx + 1}')
        qx_mask = (qx_obj.fft_freqs >= lower_freq) & (qx_obj.fft_freqs <= upper_freq)
        plt.plot(qx_obj.fft_freqs[qx_mask], qx_obj.fft_data_dB[qx_mask])
        leg.append(f'QX hardware at exp. {line_idx + 1}')
        plt.ylabel('Magnitude [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.legend(leg)
        plt.grid()
        plt.xlim([lower_freq,upper_freq])
        plt.show()

        fig_counter += 1 #update the counter
'''
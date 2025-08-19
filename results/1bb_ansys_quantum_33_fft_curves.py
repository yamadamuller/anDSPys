from framework import file_csv, file_sensor_mat, data_types
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

#Load environmental variables
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
loadp = 100 #load percentage

#Reading data for the broken bar simulation
simu_dir = "../data/1_broken_bar_110825/i_line_100%"  #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, loadp, ns, fm=fm, n_periods=600, normalize_by=np.max) # read the simulation data

#Reading data for the broken bar benchtesting
bench_dir = "../data/benchtesting_PD/experimento_2_carga_100__19200Hz_19200Hz.MAT"  #path to the directory with the bench test data
bench_test = file_sensor_mat.read(bench_dir, loadp, ns, fm=fm, n_periods=1500, normalize_by=np.max) #read the bench testing file as a batch (all files)


plt.figure(1)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experiment 100%')
plt.xlim([40,80])
plt.ylim([-80,5])
plt.ylabel('Magnitude of the current [dB]', fontsize=18)
plt.xlabel('Frequency [Hz]', fontsize=18)
#plt.title('Current Spectra')
plt.legend(leg, fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()

plt.figure(2)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experiment 33%')
plt.xlim([290,310])
plt.ylim([-120,5])
plt.ylabel('Magnitude of the current [dB]', fontsize=18)
plt.xlabel('Frequency [Hz]', fontsize=18)
#plt.title('Current Spectra')
plt.legend(leg, fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()

""""
plt.figure(2)
leg = []
plt.subplot(3,1,1)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experimental')
plt.xlim([50,70])
plt.ylim([-80,5])
plt.ylabel('FFT magnitude [dB]')
plt.title('Current Spectra')
plt.legend(leg)
plt.grid()

leg = []
plt.subplot(3,1,2)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experimental')
plt.xlim([290,310])
plt.ylim([-120,5])
plt.ylabel('FFT magnitude [dB]')
plt.legend(leg)
plt.grid()

leg = []
plt.subplot(3,1,3)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experimental')
plt.xlim([410,430])
plt.ylim([-120,5])
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.legend(leg)
plt.grid()
"""

plt.show()

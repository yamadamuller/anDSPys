from framework import file_csv, file_sensor_mat, data_types, dsp_utils
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
simu_dir = "../data/1_broken_bar_28072025"  #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, loadp, ns, fm=fm, normalize_by=np.max) # read the simulation data

#Reading data for the broken bar benchtesting
bench_dir = "../data/benchtesting_PD"  #path to the directory with the bench test data
bench_test = file_sensor_mat.read(bench_dir, loadp, ns, experiment_num=1, normalize_by=np.max) #read the bench testing file as a batch (all files)

plt.figure(1)
leg = []
#plt.subplot(3,1,1)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experimental')
plt.xlim([50,70])
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.title('Current Spectrum')
plt.legend(leg)
plt.grid()
'''
leg = []
plt.subplot(3,1,2)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_test.fft_freqs, bench_test.fft_data_dB)
leg.append('Experimental')
plt.xlim([290,310])
plt.ylim([-85,10])
plt.xlabel('Frequency [Hz]')
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
plt.ylim([-85,10])
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.legend(leg)
plt.grid()
'''
plt.show()

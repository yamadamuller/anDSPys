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
simu_dir = "../data/1_broken_bar_03082025/i_line/noR"  #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, loadp, ns, fm=fm, n_periods=600, normalize_by=np.max) # read the simulation data

#Reading data for the broken bar benchtesting
sadio_dir = "../data/sadio_110825/i_line 100%"  #path to the directory with the bench test data
simu_sadio = file_csv.read(sadio_dir, loadp, ns, fm=fm, n_periods=600, normalize_by=np.max) #read the bench testing file as a batch (all files)


plt.figure(1)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('1 broken bar')
plt.plot(simu_sadio.fft_freqs, simu_sadio.fft_data_dB)
leg.append('Healthy')
plt.xlim([40,80])
plt.ylim([-80,5])
plt.ylabel('Magnitude of the current [dB]', fontsize=18)
plt.xlabel('Frequency [Hz]', fontsize=18)
#plt.title('Current Spectra')
plt.legend(leg,fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()

plt.figure(2)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('1 broken bar')
plt.plot(simu_sadio.fft_freqs, simu_sadio.fft_data_dB)
leg.append('Healthy')
plt.xlim([290,310])
plt.ylim([-120,5])
plt.ylabel('Magnitude of the current [dB]', fontsize=18)
plt.xlabel('Frequency [Hz]', fontsize=18)
#plt.title('Current Spectra')
plt.legend(leg,fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()

""""
plt.figure(3)
leg = []
plt.subplot(3,1,1)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('1 broken bar')
plt.plot(simu_sadio.fft_freqs, simu_sadio.fft_data_dB)
leg.append('Healthy')
plt.xlim([50,70])
plt.ylim([-80,5])
plt.ylabel('FFT magnitude [dB]')
plt.title('Current Spectra')
plt.legend(leg)
plt.grid()

leg = []
plt.subplot(3,1,2)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('1 broken bar')
plt.plot(simu_sadio.fft_freqs, simu_sadio.fft_data_dB)
leg.append('Healthy')
plt.xlim([290,310])
plt.ylim([-120,5])
plt.ylabel('FFT magnitude [dB]')
plt.legend(leg)
plt.grid()

leg = []
plt.subplot(3,1,3)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('1 broken bar')
plt.plot(simu_sadio.fft_freqs, simu_sadio.fft_data_dB)
leg.append('Healthy')
plt.xlim([410,430])
plt.ylim([-120,5])
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.legend(leg)
plt.grid()
"""

plt.show()

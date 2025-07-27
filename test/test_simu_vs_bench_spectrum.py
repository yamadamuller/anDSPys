from framework import file_csv, file_sensor_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Input variables
load_percentage = 100
ns = 1800
fm = 60

#Reading data for the broken bar simulation
simu_dir = '../data/1_broken_bar_26072025' #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, load_percentage, ns, fm=fm) #read the simulation data

#Reading data for the broken bar bench test
bench_dir = '../data/benchtesting_PD' #path to the directory with the bench testing data
bench_data = file_sensor_mat.read(bench_dir, load_percentage, ns, fm=fm) #read the bench testing data

#Filter fft frequencies
simu_freq_mask = simu_data.fft_freqs>=0 #mask to filter negative frequencies
simu_data.fft_data_dB = simu_data.fft_data_dB[simu_freq_mask] #filter out the negative frequencies
simu_data.fft_freqs = simu_data.fft_freqs[simu_freq_mask] #filter out the negative frequencies
bench_freq_mask = bench_data.fft_freqs>=0 #mask to filter negative frequencies
bench_data.fft_data_dB = bench_data.fft_data_dB[bench_freq_mask] #filter out the negative frequencies
bench_data.fft_freqs = bench_data.fft_freqs[bench_freq_mask] #filter out the negative frequencies

#plot
plt.figure(1)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append('Simulation')
plt.plot(bench_data.fft_freqs, bench_data.fft_data_dB)
leg.append('Bench testing')
plt.title(f'Simu vs. Bench | 1 broken bar | load = {load_percentage} %')
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT amplitude [dB]')
plt.xlim([50,70])
plt.legend(leg)
plt.grid()
plt.show()

plt.figure(2)
leg = []
plt.plot(simu_data.i_time_grid, simu_data.i_motor)
leg.append('Simulation')
plt.plot(bench_data.time_grid, bench_data.i_r)
leg.append('Bench testing')
plt.title(f'Simu vs. Bench | 1 broken bar | load = {load_percentage} %')
plt.xlabel('Time [s]')
plt.ylabel('Current [A]')
plt.legend(leg)
plt.grid()
plt.show()
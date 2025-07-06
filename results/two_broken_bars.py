from framework import file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
wind_size = 40 #size around each component peak
ns = 1800 #synchronous speed [rpm]
fm = 60 #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the data and compute the FFT and DFT
directory = '../data/2_broken_bar/' #directory with data is located in the directory prior
healthy_directory = '../data/healthy/' #directory with data is located in the directory prior
loads = [100,75,50,25] #all the available loads to test the algorithm
fig_counter = 1 #counter to spawn new figures
proc_times = [] #list to append processing times per data

for load in loads:
    data_healthy = file_csv.read(healthy_directory, load, ns, fm) #organize the healthy output in a SimuData structure
    data = file_csv.read(directory, load, ns, fm) #organize the output in a SimuData structure

    t_init = time.time()
    peaks = dsp_utils.fft_significant_peaks(data, harm_comps, mag_threshold=-44, freq_threshold=0.2) #run the peak detection routine
    proc_times.append(time.time() - t_init)

    leg = []
    plt.figure(fig_counter)
    plt.subplot(3,1,1)
    plt.plot(data_healthy.fft_freqs[data_healthy.fft_freqs>=0], data_healthy.fft_data_dB[data_healthy.fft_freqs>=0])
    leg.append('healthy')
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[0][:,0], peaks[0][:,1], marker='x', color='black')
    leg.append('significant peaks')
    plt.title(f'Load percentage = {load}%')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([fm-int(wind_size/2), fm+int(wind_size/2)])
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(data_healthy.fft_freqs[data_healthy.fft_freqs>=0], data_healthy.fft_data_dB[data_healthy.fft_freqs>=0])
    leg.append('healthy')
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[1][:,0], peaks[1][:,1], marker='x', color='black')
    leg.append('significant peaks')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(5*fm)-int(wind_size/2), int(5*fm)+int(wind_size/2)])
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(data_healthy.fft_freqs[data_healthy.fft_freqs>=0], data_healthy.fft_data_dB[data_healthy.fft_freqs>=0])
    leg.append('healthy')
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[2][:,0], peaks[2][:,1], marker='x', color='black')
    leg.append('significant peaks')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(7*fm)-int(wind_size/2), int(7*fm)+int(wind_size/2)])
    plt.grid()

    fig_counter += 1  # increase the figure counter

print(f'Average computing time for peak detection algorithm = {np.mean(proc_times)}s')
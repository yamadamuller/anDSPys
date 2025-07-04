from framework import file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Read the data and compute the FFT and DFT
broken_directory = '../data/3_broken_bar/' #directory with data is located in the directory prior
loads = [100,75,50,25] #all the available loads to test the algorithm
fig_counter = 1 #counter to spawn new figures
amp_times = [] #list to store computing time for amplitude-based algorithm
fofd_times = [] #list to store computing time for fofd-based algorithm

#Iterative read the files and generate the output
for load in loads:
    data = file_csv.read(broken_directory, load, 1800) #read the current file linked to the load
    fft_data = dsp_utils.compute_FFT(data.i_motor) #compute the FFT
    fft_data = dsp_utils.apply_dB(fft_data) #convert from amplitude to dB
    fft_freqs = np.arange(-data.fs/2, data.fs/2, data.Res) #frequencies based on the sampling

    #Compute the sideband peaks based on the amplitude only
    t_init = time.time()
    amp_sideband_peaks = dsp_utils.sideband_peak_finder(fft_data, fft_freqs, data.slip, data.fm)  # find the sideband peaks
    amp_times.append(time.time() - t_init)

    amp_peak_points = np.array([sideband_point[0] for sideband_point in amp_sideband_peaks]) #store the peaks
    amp_freq_points = np.array([sideband_point[1] for sideband_point in amp_sideband_peaks]) #store the frequencies

    #Compute the sideband peaks based on the fdm
    t_init = time.time()
    fdm_sideband_peaks = dsp_utils.sideband_fdm_peak_finder(fft_data, fft_freqs, data.slip, data.load, data.fm)  # find the sideband peaks
    fofd_times.append(time.time() - t_init)
    fdm_peak_points = np.array([sideband_point[0] for sideband_point in fdm_sideband_peaks])  # store the peaks
    fdm_freq_points = np.array([sideband_point[1] for sideband_point in fdm_sideband_peaks])  # store the frequencies

    leg = []
    plt.figure(fig_counter)
    plt.subplot(2,1,1)
    plt.plot(fft_freqs, fft_data)
    plt.scatter(amp_freq_points, amp_peak_points, marker='x', color='black')
    leg.append('Amplitude-based')
    plt.title(f'Load percentage = {load}%')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([50, 70])
    plt.ylim([-45, 35])
    plt.grid()

    leg = []
    plt.subplot(2, 1, 2)
    plt.plot(fft_freqs, fft_data)
    plt.scatter(fdm_freq_points, fdm_peak_points, marker='x', color='black')
    leg.append('FDM-based')
    plt.title(f'Load percentage = {load}%')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([50, 70])
    plt.ylim([-45, 35])
    plt.grid()

    plt.show()

    fig_counter += 1 #increase the figure counter

print(f'Average computing time for amplitude-based algorithm = {np.mean(amp_times)}s')
print(f'Average computing time for finite difference-based algorithm = {np.mean(fofd_times)}s')
from framework import file_txt, dsp_utils, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
wind_size = 40 #size around each component peak
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the data and compute the FFT and DFT
directory = '../data/NI/i_phase/BQ_PC_3.txt' #directory with data is located in the directory prior
loads = [100] #all the available loads to test the algorithm
fig_counter = 1 #counter to spawn new figures
proc_times = [] #list to append processing times per data
all_peaks = [] #list to append all peaks registered along the loads
for load in loads:
    data = file_txt.read(directory, fm, normalize_by=np.max) #organize the output in a SimuData structure

    t_init = time.time()
    peaks = dsp_utils.fft_significant_peaks(data, harm_comps, method='distance', mag_threshold=-80, max_peaks=1, min_peak_dist=3) #run the peak detection routine
    proc_times.append(time.time() - t_init)
    all_peaks.append(peaks) #store the peaks

    leg = []
    plt.figure(fig_counter)
    plt.subplot(3,1,1)
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[0][:,0], peaks[0][:,1], marker='x', color='black')
    leg.append(f'significant peaks at {harm_comps[0]*data.fm} Hz')
    plt.title(f'Load percentage = {load}%')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([fm-int(wind_size/2), fm+int(wind_size/2)])
    plt.grid()

    leg = []
    plt.subplot(3,1,2)
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[1][:,0], peaks[1][:,1], marker='x', color='black')
    leg.append(f'significant peaks at {harm_comps[1]*data.fm} Hz')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(5*fm)-int(wind_size/2), int(5*fm)+int(wind_size/2)])
    plt.grid()

    leg = []
    plt.subplot(3,1,3)
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[2][:,0], peaks[2][:,1], marker='x', color='black')
    leg.append(f'significant peaks at {harm_comps[2]*data.fm} Hz')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(7*fm)-int(wind_size/2), int(7*fm)+int(wind_size/2)])
    plt.grid()

    fig_counter += 1  # increase the figure counter

print(f'Average computing time for peak detection algorithm = {np.mean(proc_times)}s')

output_data = dsp_utils.organize_peak_data(all_peaks, loads) #organize the output in a dataframe
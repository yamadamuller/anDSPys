from framework import file_sensor_mat, dsp_utils, data_types
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
directory = "../data/benchtesting_PD/experimento_1_carga_100__19200Hz_19200Hz.MAT" #directory with data is located in the directory prior
loads = [100] #all the available loads to test the algorithm
fig_counter = 1 #counter to spawn new figures
proc_times = [] #list to append processing times per data
all_peaks = [] #list to append all peaks registered along the loads
for load in loads:
    data = file_sensor_mat.read(directory, load, ns, fm=fm, n_periods=1600, normalize_by=np.max) #organize the output in a SimuData structure

    t_init = time.time()
    peaks = dsp_utils.fft_significant_peaks(data, harm_comps, method='distance', mag_threshold=-100, max_peaks=1) #run the peak detection routine
    proc_times.append(time.time() - t_init)
    all_peaks.append(peaks) #store the peaks

    #sidebands
    l_band = (harm_comps-2*data.slip)*fm #left sideband
    r_band = (harm_comps+2*data.slip)*fm #right sideband

    leg = []
    plt.figure(fig_counter)
    plt.subplot(3,1,1)
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{directory.split("/")[-2]}')
    plt.scatter(peaks[0][:,0], peaks[0][:,1], marker='x', color='black')
    leg.append(f'significant peaks at {harm_comps[0]*data.fm} Hz')
    plt.title(f'Load percentage = {load}%')
    plt.ylabel('Amplitude FFT [dB]')
    plt.axvline(l_band[0], linestyle='dotted', color='black')
    plt.axvline(r_band[0], linestyle='dotted', color='black')
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
    plt.axvline(l_band[1], linestyle='dotted', color='black')
    plt.axvline(r_band[1], linestyle='dotted', color='black')
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
    plt.axvline(l_band[2], linestyle='dotted', color='black')
    plt.axvline(r_band[2], linestyle='dotted', color='black')
    plt.legend(leg)
    plt.xlim([int(7*fm)-int(wind_size/2), int(7*fm)+int(wind_size/2)])
    plt.grid()

    fig_counter += 1  # increase the figure counter

print(f'Average computing time for peak detection algorithm = {np.mean(proc_times)}s')

output_data = dsp_utils.organize_peak_data(all_peaks, loads) #organize the output in a dataframe
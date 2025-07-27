from framework import file_mat, dsp_utils, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
wind_size = 40 #size around each component peak
config_file = data_types.load_config_file('../config_file.yml') #load the config file
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the data and compute the FFT and DFT
directory = '../data/labtest_1_broken_bar/' #directory with data is located in the directory prior
data = file_mat.read(directory, fm) #organize the output in a SimuData structure
all_peaks = [] #list to append all the results

t_init = time.time()
peaks = dsp_utils.fft_significant_peaks(data, harm_comps, method='distance', mag_threshold=-60, min_peak_dist=3, max_peaks=3) #run the peak detection routine
print(f'computing time for peak detection algorithm = {time.time()-t_init} s')

all_peaks.append(peaks) #update the output list

leg = []
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(data.fft_freqs, data.fft_data_dB)
leg.append(f'{directory.split("/")[-2]}')
plt.scatter(peaks[0][:, 0], peaks[0][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[0] * data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([fm - int(wind_size / 2), fm + int(wind_size / 2)])
plt.grid()

leg = []
plt.subplot(3, 1, 2)
plt.plot(data.fft_freqs, data.fft_data_dB)
leg.append(f'{directory.split("/")[-2]}')
plt.scatter(peaks[1][:, 0], peaks[1][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[1] * data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(5 * fm) - int(wind_size / 2), int(5 * fm) + int(wind_size / 2)])
plt.grid()

leg = []
plt.subplot(3, 1, 3)
plt.plot(data.fft_freqs, data.fft_data_dB)
leg.append(f'{directory.split("/")[-2]}')
plt.scatter(peaks[2][:, 0], peaks[2][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[2] * data.fm} Hz')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(7 * fm) - int(wind_size / 2), int(7 * fm) + int(wind_size / 2)])
plt.grid()

output_data = dsp_utils.organize_peak_data(all_peaks, [100]) #organize the output in a dataframe
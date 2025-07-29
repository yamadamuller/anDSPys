from framework import file_sensor_mat, file_csv, data_types, dsp_utils
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

#Important runtime variables
wind_size = 40 #size around each component peak
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the data and compute the FFT
simu_directory = '../data/1_broken_bar_28072025'
simu_data = file_csv.read(simu_directory, 100, ns, fm, normalize_by=len)
bench_directory = '../data/benchtesting_PD'
bench_data = file_sensor_mat.read(bench_directory, 100, ns, experiment_num=1, fm=fm) #organize the output in a SensorData structure
all_simu_peaks = [] #list to append all the results
all_bench_peaks = [] #list to append all the results

simu_peaks = dsp_utils.fft_significant_peaks(simu_data, harm_comps, method='distance', mag_threshold=-80, max_peaks=1) #run the peak detection routine
all_simu_peaks.append(simu_peaks) #update the output list
bench_peaks = dsp_utils.fft_significant_peaks(bench_data, harm_comps, method='distance', mag_threshold=-80, max_peaks=1) #run the peak detection routine
all_bench_peaks.append(bench_peaks) #update the output list

leg = []
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(simu_peaks[0][:, 0], simu_peaks[0][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[0] * simu_data.fm} Hz')
plt.plot(bench_data.fft_freqs, bench_data.fft_data_dB)
leg.append(f'{bench_directory.split("/")[-2]}')
plt.scatter(bench_peaks[0][:, 0], bench_peaks[0][:, 1], marker='x', color='red')
leg.append(f'significant peaks at {harm_comps[0] * bench_data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([fm - int(wind_size / 2), fm + int(wind_size / 2)])
plt.grid()

leg = []
plt.subplot(3, 1, 2)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(simu_peaks[1][:, 0], simu_peaks[1][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[1] * simu_data.fm} Hz')
plt.plot(bench_data.fft_freqs, bench_data.fft_data_dB)
leg.append(f'{bench_directory.split("/")[-2]}')
plt.scatter(bench_peaks[1][:, 0], bench_peaks[1][:, 1], marker='x', color='red')
leg.append(f'significant peaks at {harm_comps[1] * bench_data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(5 * fm) - int(wind_size / 2), int(5 * fm) + int(wind_size / 2)])
plt.grid()

leg = []
plt.subplot(3, 1, 3)
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(simu_peaks[2][:, 0], simu_peaks[2][:, 1], marker='x', color='black')
leg.append(f'significant peaks at {harm_comps[0] * simu_data.fm} Hz')
plt.plot(bench_data.fft_freqs, bench_data.fft_data_dB)
leg.append(f'{bench_directory.split("/")[-2]}')
plt.scatter(bench_peaks[2][:, 0], bench_peaks[2][:, 1], marker='x', color='red')
leg.append(f'significant peaks at {harm_comps[2] * bench_data.fm} Hz')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(7 * fm) - int(wind_size / 2), int(7 * fm) + int(wind_size / 2)])
plt.grid()

simu_output_data = dsp_utils.organize_peak_data(all_simu_peaks, [100]) #organize the output in a dataframe
bench_output_data = dsp_utils.organize_peak_data(all_bench_peaks, [100]) #organize the output in a dataframe
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
exp_nums = np.arange(1,11,1) #list the number of available experiments
plt.figure(1)
for expn in exp_nums:
    bench_directory = '../data/benchtesting_PD'
    bench_data = file_sensor_mat.read(bench_directory, 100, ns, experiment_num=expn, fm=fm, normalize_by=np.max) #organize the output in a SensorData structure
    bench_peaks = dsp_utils.fft_significant_peaks(bench_data, harm_comps, method='distance', mag_threshold=-80, max_peaks=3) #run the peak detection routine
    freq_mask = (bench_data.fft_freqs>=50)&(bench_data.fft_freqs<=70) #mask to filter the FFT data to be less resource intensive
    freqs = bench_data.fft_freqs[freq_mask] #filter the frequencies
    fft = bench_data.fft_data_dB[freq_mask] #filter the magnitudes

    leg = []
    plt.figure(1)
    plt.subplot(5, 2, int(expn))
    plt.plot(freqs, fft)
    leg.append(f'Experiment = {expn}')
    plt.scatter(bench_peaks[0][:, 0], bench_peaks[0][:, 1], marker='x', color='red')
    leg.append(f'significant peaks at {harm_comps[0] * bench_data.fm} Hz')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([50,70])
    plt.grid()
    '''
    leg = []
    plt.subplot(3, 1, 2)
    #plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
    #leg.append(f'{simu_directory.split("/")[-2]}')
    #plt.scatter(simu_peaks[1][:, 0], simu_peaks[1][:, 1], marker='x', color='black')
    #leg.append(f'significant peaks at {harm_comps[1] * simu_data.fm} Hz')
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
    #plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
    #leg.append(f'{simu_directory.split("/")[-2]}')
    #plt.scatter(simu_peaks[2][:, 0], simu_peaks[2][:, 1], marker='x', color='black')
    #leg.append(f'significant peaks at {harm_comps[0] * simu_data.fm} Hz')
    plt.plot(bench_data.fft_freqs, bench_data.fft_data_dB)
    leg.append(f'{bench_directory.split("/")[-2]}')
    plt.scatter(bench_peaks[2][:, 0], bench_peaks[2][:, 1], marker='x', color='red')
    leg.append(f'significant peaks at {harm_comps[2] * bench_data.fm} Hz')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(7 * fm) - int(wind_size / 2), int(7 * fm) + int(wind_size / 2)])
    plt.grid()
    '''
plt.show()

#bench_output_data = dsp_utils.organize_peak_data(all_bench_peaks, [100]) #organize the output in a dataframe
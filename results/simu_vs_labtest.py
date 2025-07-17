from framework import file_mat, file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
wind_size = 40 #size around each component peak
fm = 60 #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the lab data and compute the FFT and DFT
lab_directory = '../data/labtest_1_broken_bar/' #directory with data is located in the directory prior
lab_data = file_mat.read(lab_directory, fm) #organize the output in a SimuData structure
lab_all_peaks = [] #list to append all the results
lab_peaks = dsp_utils.fft_significant_peaks(lab_data, harm_comps, method='distance', mag_threshold=-60, min_peak_dist=3, max_peaks=3) #run the peak detection routine
lab_all_peaks.append(lab_peaks)

#Read the simulation data and compute the FFT and DFT
simu_directory = '../data/1_broken_bar/' #directory with data is located in the directory prior
simu_data = file_csv.read(simu_directory, 100, 1800, fm) #organize the healthy output in a SimuData structure
simu_all_peaks = [] #list to append all the results
simu_peaks = dsp_utils.fft_significant_peaks(simu_data, harm_comps, method='distance', mag_threshold=-60, max_peaks=3) #run the peak detection routine
simu_all_peaks.append(simu_peaks)

#Plot the data
leg = []
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(lab_data.fft_freqs, lab_data.fft_data_dB)
leg.append(f'{lab_directory.split("/")[-2]}')
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(lab_peaks[0][:,0], lab_peaks[0][:,1], marker='x', color='red')
leg.append(f'significant lab testing peaks at {harm_comps[0]*lab_data.fm} Hz')
plt.scatter(simu_peaks[0][:,0], simu_peaks[0][:,1], marker='x', color='black')
leg.append(f'significant simulation peaks at {harm_comps[0]*simu_data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([fm - int(wind_size / 2), fm + int(wind_size / 2)])
plt.title('Lab vs Simu | 1 broken bar')
plt.grid()

leg = []
plt.subplot(3, 1, 2)
plt.plot(lab_data.fft_freqs, lab_data.fft_data_dB)
leg.append(f'{lab_directory.split("/")[-2]}')
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(lab_peaks[1][:,0], lab_peaks[1][:,1], marker='x', color='red')
leg.append(f'significant lab testing peaks at {harm_comps[1]*lab_data.fm} Hz')
plt.scatter(simu_peaks[1][:,0], simu_peaks[1][:,1], marker='x', color='black')
leg.append(f'significant simulation peaks at {harm_comps[1]*simu_data.fm} Hz')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(5 * fm) - int(wind_size / 2), int(5 * fm) + int(wind_size / 2)])
plt.grid()

leg = []
plt.subplot(3, 1, 3)
plt.plot(lab_data.fft_freqs, lab_data.fft_data_dB)
leg.append(f'{lab_directory.split("/")[-2]}')
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
leg.append(f'{simu_directory.split("/")[-2]}')
plt.scatter(lab_peaks[2][:,0], lab_peaks[2][:,1], marker='x', color='red')
leg.append(f'significant lab testing peaks at {harm_comps[2]*lab_data.fm} Hz')
plt.scatter(simu_peaks[2][:,0], simu_peaks[2][:,1], marker='x', color='black')
leg.append(f'significant simulation peaks at {harm_comps[2]*simu_data.fm} Hz')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([int(7 * fm) - int(wind_size / 2), int(7 * fm) + int(wind_size / 2)])
plt.grid()

#Organized outputs
simu_output_data = dsp_utils.organize_peak_data(simu_all_peaks, [100]) #organize the output in a dataframe
lab_output_data = dsp_utils.organize_peak_data(lab_all_peaks, [100]) #organize the output in a dataframe
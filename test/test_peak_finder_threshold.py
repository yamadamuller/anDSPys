from framework import file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Read the data and compute the FFT and DFT
directory = '../data/broken/' #directory with data is located in the directory prior
data = file_csv.read(directory, 100, 1800, 60) #organize the output in a SimuData structure
fft_data = dsp_utils.compute_FFT(data.i_motor) #compute the FFT
fft_data = dsp_utils.apply_dB(fft_data) #convert from amplitude to dB
fft_freqs = np.arange(-data.fs/2, data.fs/2, data.Res) #frequencies based on the sampling
sideband_peaks = dsp_utils.sideband_peak_finder(fft_data, fft_freqs, data.slip, data.fm) #find the sideband peaks
peak_points = np.array([sideband_point[0] for sideband_point in sideband_peaks]) #store the peaks
freq_points = np.array([sideband_point[1] for sideband_point in sideband_peaks]) #store the frequencies

leg = []
plt.figure(1)
plt.plot(fft_freqs, fft_data)
leg.append('One broken bar')
plt.scatter(freq_points, peak_points, marker='x', color='black')
leg.append('Peaks')
plt.title('Frequency domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude FFT [dB]')
plt.legend(leg)
plt.xlim([50,70])
plt.ylim([-45,35])
plt.grid()
plt.show()
from framework import file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Read the data and compute the FFT and DFT
directory = '../data/healthy/' #directory with data is located in the directory prior
data = file_csv.read(directory, 100, 1800, 60) #organize the output in a SimuData structure
fft_data = dsp_utils.compute_FFT(data.i_motor, normalize_by=np.max) #compute the FFT of the current (shifted + normalized)
dft_data_r, dft_data_i = dsp_utils.compute_DFT(data.i_motor, np.zeros_like(data.i_motor)) #compute the FFT of the current (shifted + normalized)
dft_data = np.abs(dft_data_r+dft_data_i)/np.max(np.abs(dft_data_r+dft_data_i)) #normalized magnitude of the DFT
fourier_freqs = np.arange(-data.fs/2, data.fs/2, data.Res) #frequencies based on the sampling

#Filter for frequencies around the fundamental
freq_mask = (fourier_freqs>=50)&(fourier_freqs<=80) #mask to filter the frequencies

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(data.time_grid[freq_mask], data.i_motor[freq_mask])
plt.xlabel('t (s)')
plt.ylabel('Current [A]')

plt.subplot(3,1,2)
plt.plot(fourier_freqs[freq_mask], fft_data[freq_mask])
plt.xlabel('f (Hz)')
plt.ylabel('FFT magnitude')

plt.subplot(3,1,3)
plt.plot(fourier_freqs[freq_mask], dft_data[freq_mask])
plt.xlabel('f (Hz)')
plt.ylabel('DFT magnitude')

plt.show()
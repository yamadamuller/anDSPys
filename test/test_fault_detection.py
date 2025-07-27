from framework import file_mat, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def apply_movavg(signal, kernel_size):
    return np.convolve(signal, np.ones(kernel_size)/kernel_size, mode='same')

def apply_movstd(signal, samples):
    idx = np.arange(samples, len(signal),1)
    mov_std = np.zeros((len(signal),))
    signal = np.pad(signal, (samples-1,0)) #pad the signal
    for i in idx:
        mov_std[int(i-samples)] = np.std(signal[i-samples:i+1])
    return mov_std

#Important runtime variables
wind_size = 40 #size around each component peak
ns = 1800 #synchronous speed [rpm]
fm = 60 #fundamental frequency
harm_comps = [1,5,7] #harmonic components
mags = [-30,-44,-44] #all the magnitude thresholds

#Read the data and compute the FFT and DFT
directory = '../data/labtest_1_broken_bar/' #directory with data is located in the directory prior
data = file_mat.read(directory, fm) #organize the output in a SimuData structure
peaks = dsp_utils.fft_significant_peaks(data, harm_comps, method='distance', mag_threshold=-60, min_peak_dist=3, max_peaks=1) #run the peak detection routine

# Filter only positive values from the fft frequencies
freq_mask = data.fft_freqs >= 0  # mask to filter negative frequencies
data.fft_freqs = data.fft_freqs[freq_mask]  # filtered frequencies
data.fft_data_amp = data.fft_data_amp[freq_mask]  # filtered magnitudes amplitude
data.fft_data_dB = data.fft_data_dB[freq_mask]  # filtered magnitudes dB
bound_idx = 524
harm_idx = 1274
pk_idx = 0
wind_freqs = data.fft_freqs[bound_idx:harm_idx]
wind_spectrum = data.fft_data_dB[bound_idx:harm_idx]

#moving filters
gamma = 1.25
mov_avg = np.convolve(wind_spectrum, np.ones(9)/9, mode='same')
sub = np.abs(wind_spectrum-mov_avg)
mov_std = gamma*dsp_utils.apply_moving_filter(wind_spectrum, np.std,9)

#fault detection function
h_peaks = peaks[pk_idx]
peak_idx = np.isin(wind_freqs, h_peaks[:,0]).nonzero()[0] #find the index of every peak in the frequency window
adaptive_threshold = sub[peak_idx]>=mov_std[peak_idx] #mask to find where the peaks overflow the threshold
fault_occ = np.int32(adaptive_threshold)*peak_idx #multiply the mask with the indexes (if false the index will go to zero)
fault_idx = fault_occ[fault_occ>0]
fault_fun = np.zeros_like(wind_spectrum) #benchmark function
fault_fun[fault_idx] = 1 #detected flaw at the indexes

#plots
plt.subplot(2,1,1)
leg = []
plt.plot(wind_freqs, wind_spectrum)
leg.append('Spectrum')
plt.plot(wind_freqs, mov_std, linewidth=0.8)
leg.append('mov_std')
plt.plot(wind_freqs, sub, linewidth=1)
leg.append('|spectrum-mov_avg|')
coords = np.arange(0,3,1)
for i in range(len(coords)):
    plt.axvline(peaks[pk_idx][coords[i],0],color='red', linewidth=0.5)
leg.append('peaks')
plt.legend(leg)
plt.xlim([50,70])
plt.subplot(2,1,2)
leg = []
plt.plot(wind_freqs, fault_fun)
leg.append('fault benchmark')
plt.legend(leg)
plt.xlim([50,70])
plt.show()
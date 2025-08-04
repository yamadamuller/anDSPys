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
leg = []
for expn in exp_nums:
    bench_directory = '../data/benchtesting_PD'
    bench_data = file_sensor_mat.read(bench_directory, 100, ns, experiment_num=expn, fm=fm, normalize_by=len) #organize the output in a SensorData structure
    freq_mask = (bench_data.fft_freqs>=50)&(bench_data.fft_freqs<=70) #mask to filter the FFT data to be less resource intensive
    fft = bench_data.fft_data_dB[freq_mask]
    freqs = bench_data.fft_freqs[freq_mask]
    plt.plot(freqs, fft)
    leg.append(f'Experiment = {expn}')
    plt.xlim([50,70])

plt.ylabel('Amplitude FFT [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('QuantumX')
plt.legend(leg)
plt.grid()
plt.show()


#bench_output_data = dsp_utils.organize_peak_data(all_bench_peaks, [100]) #organize the output in a dataframe
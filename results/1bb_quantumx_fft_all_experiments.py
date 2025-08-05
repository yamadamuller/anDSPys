from framework import file_sensor_mat, data_types
import os
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
bench_directory = '../data/benchtesting_PD'
bench_list = [os.path.join(bench_directory,f) for f in os.listdir(bench_directory) if '100' in f] #create a list with .MAT files
bench_list.sort() #sort the list
bench_list.append(bench_list[0]) #add the 10th experiment to the end of the list
bench_list.pop(0) #remove the 10th experiment from the beginning of the list
bench_obj = file_sensor_mat.read(bench_list, 100, ns, fm=fm, n_periods=1200, batch=True, normalize_by=np.max) #organize the output in a SensorData structure

plt.figure(1)
leg = []
for batch_idx in range(len(bench_obj.batch_data)):
    batch_elem = bench_obj.batch_data[batch_idx] #extract a SensorData from the batch_dara array
    freq_mask = (batch_elem.fft_freqs>=50)&(batch_elem.fft_freqs<=70) #mask to filter the FFT data to be less resource intensive
    fft = batch_elem.fft_data_dB[freq_mask]
    freqs = batch_elem.fft_freqs[freq_mask]
    plt.plot(freqs, fft)
    leg.append(f'Experiment = {batch_idx}')
    plt.xlim([50,70])

plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('Current spectra from all experiments')
plt.legend(leg)
plt.grid()
plt.show()

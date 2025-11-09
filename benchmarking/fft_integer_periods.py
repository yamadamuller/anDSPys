from framework import file_csv, dsp_utils, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency

#Read the data and compute the FFT
directory = '../data/1_broken_bar_18082025/' #directory with data is located in the directory prior
periods = [25, 50, 100, 200, 400, 600]
plt.figure(1)
leg = []
for n in periods:
    data = file_csv.read(directory, 100, ns, fm, n_periods=n, normalize_by=np.max) #organize the output in a SimuData structure
    plt.plot(data.fft_freqs, data.fft_data_dB)
    leg.append(f'{n} periods')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.xlim([40, 80])
plt.ylim([-90, 5])
plt.legend(leg)
plt.grid()
plt.show()
from framework import file_csv, dsp_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

#Important runtime variables
wind_size = 40 #size around each component peak
ns = 1800 #synchronous speed [rpm]
fm = 60 #fundamental frequency
harm_comps = [1,5,7] #harmonic components

#Read the data and compute the FFT and DFT
directory = '../data/1_broken_bar/' #directory with data is located in the directory prior
healthy_directory = '../data/healthy/' #directory with data is located in the directory prior
loads = [100,75,50,25] #all the available loads to test the algorithm
fig_counter = 1 #counter to spawn new figures
proc_times = [] #list to append processing times per data

for load in loads:
    data_healthy = file_csv.read(healthy_directory, load, ns, fm) #organize the healthy output in a SimuData structure
    data = file_csv.read(directory, load, ns, fm) #organize the output in a SimuData structure

    t_init = time.time()
    iqms_left = dsp_utils.fft_IQM_per_harmonic(data, harm_comps) #run the peak detection routine
    iqms_right = dsp_utils.fft_IQM_per_harmonic(data, harm_comps, spike_side='right') #run the peak detection routine
    proc_times.append(time.time() - t_init)

    leg = []
    plt.figure(fig_counter)
    plt.subplot(3,1,1)
    plt.plot(data.fft_freqs, data.fft_data_dB, color='tab:orange')
    leg.append(f'{directory.split("/")[-2]}')
    plt.axhline(iqms_left[2][0], xmax=harm_comps[2] * data.fm, color='black', linestyle='dashed')
    leg.append(f'Lower IQM boundary')
    plt.axhline(iqms_left[2][1], xmax=harm_comps[2] * data.fm, color='black', linestyle='dashed')
    leg.append(f'Upper left IQM boundary')
    plt.axhline(iqms_right[0][0], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Lower right IQM boundary')
    plt.axhline(iqms_right[0][1], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Upper right IQM boundary')
    plt.title(f'Load percentage = {load}%')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([fm-int(wind_size/2), fm+int(wind_size/2)])
    plt.grid()

    leg = []
    plt.subplot(3,1,2)
    plt.plot(data.fft_freqs, data.fft_data_dB, color='tab:orange')
    leg.append(f'{directory.split("/")[-2]}')
    plt.axhline(iqms_left[2][0], xmax=harm_comps[2] * data.fm, color='black', linestyle='dashed')
    leg.append(f'Lower IQM boundary')
    plt.axhline(iqms_left[2][1], xmax=harm_comps[2] * data.fm, color='black', linestyle='dashed')
    leg.append(f'Upper left IQM boundary')
    plt.axhline(iqms_right[0][0], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Lower right IQM boundary')
    plt.axhline(iqms_right[0][1], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Upper right IQM boundary')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(5*fm)-int(wind_size/2), int(5*fm)+int(wind_size/2)])
    plt.grid()

    leg = []
    plt.subplot(3,1,3)
    plt.plot(data.fft_freqs, data.fft_data_dB, color='tab:orange')
    leg.append(f'{directory.split("/")[-2]}')
    plt.axhline(iqms_left[2][0], xmax=harm_comps[2]*data.fm, color='black', linestyle='dashed')
    leg.append(f'Lower IQM boundary')
    plt.axhline(iqms_left[2][1], xmax=harm_comps[2]*data.fm, color='black', linestyle='dashed')
    leg.append(f'Upper left IQM boundary')
    plt.axhline(iqms_right[0][0], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Lower right IQM boundary')
    plt.axhline(iqms_right[0][1], xmax=harm_comps[0] * data.fm, color='red', linestyle='dashed')
    leg.append(f'Upper right IQM boundary')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude FFT [dB]')
    plt.legend(leg)
    plt.xlim([int(7*fm)-int(wind_size/2), int(7*fm)+int(wind_size/2)])
    plt.grid()

    fig_counter += 1  # increase the figure counter

print(f'Average computing time for peak detection algorithm = {np.mean(proc_times)}s')
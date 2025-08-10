from framework import file_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file = '../data/narco/struct_r1b_R1.mat' #directory with data is located in the directory prior
experiment = 5 #number of the experiment
n_periods = 900 #number of integer periods to extract from the currents
step = 0.5 #torque step
torque = 4 #N.m
data = file_mat.read(file, torque, n_periods=n_periods, exp_num=experiment, normalize_by=np.max) #organize the output in a LaipseData structure

#plot
plt.figure(1)
leg = []
plt.plot(data.fft_freqs, data.fft_data_dB)
leg.append(f'Torque = {torque} N.m')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amp. [dB]')
plt.title(f'[Exp. {experiment}] Current spectrum')
plt.legend(leg)
plt.xlim([40,80])
plt.ylim([-100,5])
plt.grid()
plt.show()
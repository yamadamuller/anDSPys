'''
This is an example file to guide on processing sensor (bench test) output data into the anDSPys framework.
'''

#import the minimum required packages
from framework import file_sensor_mat, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Load some environmental variables stored in the configuration yml file ('../config_file.yml').
#Although this is not an obligatory step, it loads some variables that may be repeated throughout multiple scripts
#(i.e., fundamental frequency).
#If one changes them between testings, it is not necessary to manually update the values in all scripts.
config_file = data_types.load_config_file('../config_file.yml') #load the config file
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed

#The function responsible for loading the .MAT sensor data output belongs to the file_sensor_mat package.
#Call it with the required arguments and store the output into a variable.
data_directory = '../data/benchtesting_PD' #define the directory where your .csv data is located
load = 100 #define which simulation you want to load based on the load percentage
n_exp = 3 #define which experiment you want to load based on its number
#For more information on how to format the data, see the README file in the "Running the framework" section
try:
    data = file_sensor_mat.read(data_directory, load, ns,
                                experiment_num=n_exp, fm=fm) #run the file reading function
except Exception as e:
    raise RuntimeError(f'[read_sensor_data] Read function failed with {e}')
#Optional arguments:
#-> fm: fundamental frequency (60 Hz by default)
#-> experiment_number: 1 by default
#-> transient: flag to filter out, or not, the transient in the electrical signals (False by default = Filter out)
#-> batch: flag to load all the sensor data in a batch (False by default)
#-> normalize_by: pass which function will be used to normalize the FFT (len, np.max, ...)

#The assigned variable should be a SensorData structure that compiles the most important data from the testing as attributes
if type(data) == data_types.SensorData:
    print(type(data))
else:
    raise TypeError(f'[read_sensor_data] The variable is not a SensorData structure, data type = {type(data)}!')
#For more information on the SimuData structure, see the "../framework/data_types.py" script

#You may now access the structure's attributes to process your data into the framework.
print(f'SensorData available attributes: {data.__dict__.keys()}')

#For instance, load the spectrum magnitude of the current in dB, stored in "fft_data_dB", along with its frequencies.
fft_data = data.fft_data_dB #FFT magnitude in dB
fft_freqs = data.fft_freqs #FFT frequencies

#Optional step: The frequencies computed by the framework contain negative values.
#You can create a mask to filter them out, facilitating visualization.
freq_mask = fft_freqs >= 0
fft_data_filt = fft_data[freq_mask]
fft_freqs_filt = fft_freqs[freq_mask]

#Create a plot of the FFT
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(fft_freqs, fft_data)
plt.ylabel('FFT magnitude [dB]')
plt.title('FFT w/ negative frequencies')
plt.grid()

plt.subplot(3,1,2)
plt.plot(fft_freqs_filt, fft_data_filt)
plt.ylabel('FFT magnitude (dB)')
plt.title('FFT w/o negative frequencies')
plt.grid()
plt.show()

plt.subplot(3,1,3)
plt.plot(fft_freqs_filt, fft_data_filt)
plt.xlim([50,70]) #reduce the x-axis to improve visualization
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude (dB)')
plt.title('FFT zoomed and centered at 60 Hz')
plt.grid()
plt.show()

#Or, you might want to process only the current in the time domain
i_data = data.i_r #current of the R-phase in A
t_data = data.time_grid #time in s
plt.figure(2)
plt.plot(t_data, i_data)
plt.xlim([10, 10.5]) #reduce the x-axis to improve visualization
plt.ylabel('Current [A]')
plt.xlabel('Time [s]')
plt.title('Motor current (R-phase) in time domain')
plt.grid()
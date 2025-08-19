'''
This is an example file to guide on processing ANSYS output data into the anDSPys framework.
'''

#import the minimum required packages
from framework import file_csv, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Load some environmental variables stored in the configuration yml file ('../config_file.yml').
#Although this is not an obligatory step, it loads some variables that may be repeated throughout multiple scripts
#(i.e., fundamental frequency).
#If one changes them between simulations, it is not necessary to manually update the values in all scripts.
config_file = data_types.load_config_file('../config_file.yml') #load the config file
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed

#The function responsible for loading the .csv data output from ANSYS belongs to the file_csv package.
#Call it with the required arguments and store the output into a variable.
data_directory = '../data/1_broken_bar' #define the directory where your .csv data is located
load = 100 #define which simulation you want to load based on the load percentage
n_periods = 600 #define how many peroids will be extracted from the current signal
#For more information on how to format the data, see the README file in the "Running the framework" section
try:
    data = file_csv.read(data_directory, load, ns, fm=fm, n_periods=n_periods, normalize_by=np.max) #run the file reading function
except Exception as e:
    raise RuntimeError(f'[read_ansys_data] Read function failed with {e}')
#Optional arguments:
#-> fm: fundamental frequency (60 Hz by default)
#-> transient: flag to filter out the transient in the signal (False by default)
#-> normalize_by: pass which function will be used to normalize the FFT (len, np.max, ...)

#The assigned variable should be a SimuData structure that compiles the most important data from the simulation as attributes
if type(data) == data_types.SimuData:
    print(type(data))
else:
    raise TypeError(f'[read_ansys_data] The variable is not a SimuData structure, data type = {type(data)}!')
#For more information on the SimuData structure, see the "../framework/data_types.py" script

#You may now access the structure's attributes to process your data into the framework.
print(f'SimuData available attributes: {data.__dict__.keys()}')

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
i_data = data.i_motor #current in A
t_data = data.time_grid #time in s
plt.figure(2)
plt.plot(t_data, i_data)
plt.xlim([4,4.5]) #reduce the x-axis to improve visualization
plt.ylabel('Current [A]')
plt.xlabel('Time [s]')
plt.title('Motor current in time domain')
plt.grid()
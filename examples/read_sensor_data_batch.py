'''
This is an example file to guide on processing sensor (bench test) output batch data into the anDSPys framework.
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
#If one changes them between simulations, it is not necessary to manually update the values in all scripts.
config_file = data_types.load_config_file('../config_file.yml') #load the config file
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed

#The function responsible for loading the .MAT sensor data output in a batch belongs to the file_sensor_mat package.
#Call it with the required arguments and store the output into a variable.
data_directory = '../data/benchtesting_PD' #define the directory where your .csv data is located
load = 100 #define which simulation you want to load based on the load percentage
#For more information on how to format the data, see the README file in the "Running the framework" section
try:
    data = file_sensor_mat.read(data_directory, load, ns,
                                fm=fm, batch=True) #run the file reading function
except Exception as e:
    raise RuntimeError(f'[read_sensor_data] Read function failed with {e}')
#Optional arguments:
#-> fm: fundamental frequency (60 Hz by default)
#-> transient: flag to filter out, or not, the transient in the electrical signals (False by default = Filter out)
#-> normalize_by: pass which function will be used to normalize the FFT (len, np.max, ...)

#The assigned variable should be a BatchSensorData structure.
if type(data) == data_types.BatchSensorData:
    print(type(data))
else:
    raise TypeError(f'[read_sensor_data] The variable is not a SensorData structure, data type = {type(data)}!')
#For more information on the SimuData structure, see the "../framework/data_types.py" script

#You may now access the structure's attributes to process your data into the framework.
print(f'SensorData available attributes: {data.__dict__.keys()}')

#This structure compiles all the SensorData structures from the available experiments in an array.
#Each element of the array corresponds to the respective experiment number subtracted by 1 (python has 0-based indexing)
#i.e., experiment 1 SensorData is stored at the element 0 of the object.batch_data attribute, and so on...

#For instance, load the all the spectrum magnitudes of the currents in dB, stored in each fft_data_dB", along with its frequencies.
plt.figure(1)
leg = [] #create a list to append the legends per data
for batch_idx in range(len(data.batch_data)):
    batch_elem = data.batch_data[batch_idx] #current processed element
    freq_mask = (batch_elem.fft_freqs>=50)&(batch_elem.fft_freqs<70) #mask the frequencies between 50 and 60 Hz to reduce samples (less resource-intensive)
    plt.plot(batch_elem.fft_freqs[freq_mask], batch_elem.fft_data_dB[freq_mask])
    leg.append(f'Experiment {batch_idx}')
plt.xlim([50, 70]) #reduce the x-axis to improve visualization
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.title('Batch data current spectrum')
plt.legend(leg)
plt.grid()
plt.show()

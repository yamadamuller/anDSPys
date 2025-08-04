import os
from framework import data_types
from scipy.io import loadmat
import numpy as np

def read(file, load_percentage, ns, fm=60, n_periods=None, transient=False, batch=False, normalize_by=np.max):
    '''
    :param file: path to the .MAT output file in the local filesystem (or a list of files for batch=True)
    :param load_percentage: percentage of the load used in the simulation [%]
    :param ns: synchronous speed of the simulated motor [rpm]
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
    :param transient: flag to filter out the transient
    :param batch: compute the data as the average from multiple experiments
    :param normalize_by: which function will be used to normalize the FFT
    return SensorData structure with the required data from the lab testing
    '''
    #Check in case of batch
    if not batch:
        #Check if the directory passed as argument exists
        if not os.path.isfile(file):
            raise ValueError(f'[file_sensor_mat] File {file} does not exist!')
        raw_data = loadmat(file) #read the raw .MAT file into dictionary format
        data = data_types.SensorData(raw_data, ns, fm=fm, n_periods=n_periods, transient=transient,
                                     normalize_by=normalize_by) #create the data structure with the lab tested output
    else:
        if not type(file)==list:
            raise TypeError(f'[file_sensor_mat] batch={batch} requires a list of files as input!')

        data = data_types.BatchSensorData(file, load_percentage, ns, fm=fm, n_periods=n_periods, transient=transient,
                                          normalize_by=normalize_by) #compile the batch of data available

    return data
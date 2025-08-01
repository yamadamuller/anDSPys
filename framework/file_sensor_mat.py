import os
from framework import data_types
from scipy.io import loadmat

def read(filedir, load_percentage, ns, experiment_num=None, fm=60, transient=False, batch=False, normalize_by=len):
    '''
    :param filedir: path to the .csv output file in the local filesystem
    :param load_percentage: percentage of the load used in the simulation [%]
    :param ns: synchronous speed of the simulated motor [rpm]
    :param experiment_num: number of the experiment (None by default -> exp. 1)
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :param transient: flag to filter out the transient
    :param batch: compute the data as the average from multiple experiments
    :param normalize_by: which function will be used to normalize the FFT
    return SensorData structure with the required data from the lab testing
    '''
    #Check if the directory passed as argument exists
    if not os.path.isdir(filedir):
        raise ValueError(f'[file_sensor_mat] Directory {filedir} does not exist!')

    if not batch:
        if experiment_num is None:
            experiment_num = 1 #defaults to 1

        #process the raw data for current
        sensor_file = os.path.join(filedir, f'experimento_{int(experiment_num)}_carga_{load_percentage}__19200Hz_19200Hz.MAT') #define which file to read from

        #Check if the file exists
        if not os.path.isfile(sensor_file):
            raise ValueError(f'[file_sensor_mat] File {sensor_file} does not exist!')

        raw_data = loadmat(sensor_file) #read the raw .MAT file into dictionary format
        data = data_types.SensorData(raw_data, ns, fm, transient, normalize_by=normalize_by) #create the data structure with the lab tested output
    else:
        data = data_types.BatchSensorData(filedir, load_percentage, ns, fm, transient, normalize_by=normalize_by) #compile the batch of data available

    return data
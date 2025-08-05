import pandas as pd
import os
from framework import data_types
import numpy as np

def read(filedir, load_percentage, ns, fm=60, n_periods=None, transient=False, normalize_by=np.max):
    '''
    :param filedir: path to the .csv output file in the local filesystem
    :param load_percentage: percentage of the load used in the simulation [%]
    :param ns: synchronous speed of the simulated motor [rpm]
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
    :param transient: flag to filter out the transient in the signal (False by default)
    :param normalize_by: which function will be used to normalize the FFT
    :return: SimuData structure with the required data from the simulation
    '''
    #Check if the directory passed as argument exists
    if not os.path.isdir(filedir):
        raise ValueError(f'[file_csv] Directory {filedir} does not exist!')

    #Validate if the load_percentage is valid
    load_perc = float(load_percentage) #convert to load
    if not (load_perc>=0) and not (load_percentage<=100):
        raise ValueError(f'[file_csv] Load percentage should be 0<=value<=100')

    #process the raw data for current
    current_file = os.path.join(filedir, f'corrente {int(load_perc)}.csv') #define which file to read from

    #Check if the file exists
    if not os.path.isfile(current_file):
        print(f'[file_csv] File {current_file} does not exist, compiling the data without it!')
        current_data = None #load the raw data as None
    else:
        current_data = pd.read_csv(current_file).to_numpy() #read the raw csv file and convert to numpy

    #process the raw data for speed
    speed_file = os.path.join(filedir, f'velocidade {int(load_perc)}.csv')  # define which file to read from

    #Check if the file exists
    if not os.path.isfile(speed_file):
        print(f'[file_csv] File {speed_file} does not exist, compiling data without it!')
        speed_data = None #load the raw data as None
    else:
        speed_data = pd.read_csv(speed_file).to_numpy()  #read the raw csv file and convert to numpy

    data = data_types.SimuData(current_data, speed_data, int(load_percentage), ns, fm=fm, n_periods=n_periods,
                               transient=transient, normalize_by=normalize_by) #create the data structure with the simulation output
    return data

import pandas as pd
import os
from framework import data_types
from scipy.io import loadmat

def read(filedir, fm=60):
    '''
    :param filedir: path to the .mat output file in the local filesystem
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :return: SimuData structure with the required data from the simulation
    '''
    #Check if the directory passed as argument exists
    if not os.path.isdir(filedir):
        raise ValueError(f'[file_m] Directory {filedir} does not exist!')

    #process the raw data for current
    current_file = os.path.join(filedir, f'corrente.mat') #define which file to read from

    #Check if the file exists
    if not os.path.isfile(current_file):
        raise ValueError(f'[file_m] File {current_file} does not exist!')

    raw_data = loadmat(current_file) #read the raw .mat file and convert to numpy
    data_keys = list(raw_data.keys()) #list all the dictionary keys from the .mat structure
    raw_data = raw_data[data_keys[-1]] #extract only the part with the matrix
    data = data_types.LabData(raw_data, fm) #create the data structure with the lab tested output
    return data
import os
from framework import data_types
import numpy as np

def read(file, fm=60, batch=False, normalize_by=np.max):
    '''
    :param file: path to the .txt output file in the local filesystem
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :param batch: compute the data as the average from multiple experiments
    :param normalize_by: which function will be used to normalize the FFT
    :return: NIHardwareData structure with the required data from the lab testing
    '''
    #Check in case of batch file is a list
    if batch:
        if not type(file)==list:
            raise TypeError(f'[file_txt] batch={batch} requires a list of files as input!')
    else:
        #Check if the file passed as argument exists
        if not os.path.isfile(file):
            raise ValueError(f'[file_txt] File {file} does not exist!')

    if not batch:
        data = data_types.NIHardwareData(file, fm=fm, normalize_by=normalize_by) #create the data structure with the hardware output
    else:
        data = data_types.BatchNIHardwareData(file, fm=fm, normalize_by=normalize_by) #compile the batch of data available

    return data
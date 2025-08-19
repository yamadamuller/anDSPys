import os
from framework import data_types
from scipy.io import loadmat
import numpy as np

def read(file, torque, fs=50e3, fm=60, n_periods=None, exp_num=1, transient=False, normalize_by=np.max):
    '''
    :param file: path to the .MAT output file in the local filesystem
    :param torque: torque of the experiment to extract the data
    :param fs: sampling frequency [Hz] (50.05 kHz by default)
    :param fm: fundamental frequency [Hz] (60 Hz by default)
    :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
    :param exp_num: which experiment will be extracted from the experimental data (1 by default)
    :param transient: flag to filter out the transient (False by default->no transient)
    :param normalize_by: which function will be used to normalize the FFT
    :return LaipseData structure with the required data from the lab testing
    '''
    #Check if the directory passed as argument exists
    if not os.path.isfile(file):
        raise ValueError(f'[file_mat] File {file} does not exist!')

    data = data_types.LaipseData(file, torque, fs=fs, fm=fm, n_periods=n_periods, exp_num=exp_num, transient=transient,
                                 normalize_by=normalize_by) #create the data structure with the LAIPSE tested output
    return data
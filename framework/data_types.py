import os
import numpy as np
from framework import electromag_utils, dsp_utils
from scipy.io import loadmat
import yaml
import datetime

def load_config_file(config_file):
    '''
    :param config_file: path to the config ymal file
    :return: the config file data
    '''
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f'[data_types] {config_file} does not exist!')
    if not config_file.endswith('.yml'):
        raise ValueError(f'[data_types] config_file must be a .yml file')

    #Load the configurations
    try:
        with open(config_file, 'rt') as f:
            config =yaml.safe_load(f.read()) #load the config file
        return config
    except:
        raise RuntimeError(f'[data_types] Failed to load {config_file}!')

def read_NIHW_data(file):
    '''
    :param file: path to the .txt output file in the local filesystem
    :return: the data from the .txt file in numpy.array format (relative timestamp and current value)
    '''
    data = [] #list to append all the computed samples from each line
    ref_time = None #store the timestamp of the first sample to subtract from in the next timestamps
    timestamp_flag = True #flag to monitor the first sample's timestamp
    with open(file) as infile: #read the .txt w/o loading into memory
        for line in infile:
            line_att = line.split() #list all the attributes per sample separately
            line_att[1] = line_att[1].replace(',', '.') #timestamp with '.' decimal notation
            line_att[2] = line_att[2].replace(',', '.') #current with '.' decimal notation
            fmt_time_sample = datetime.datetime.strptime(line_att[1], '%H:%M:%S.%f') #format into datetime object

            #Handle reference time attribute
            if timestamp_flag:
                ref_time = fmt_time_sample #commute the variables
                timestamp_flag = False #set flag as False to avoid overwriting the variable

            rel_timestamp = fmt_time_sample - ref_time #extract the relative timestamp
            rel_timestamp = float(rel_timestamp.seconds)+float(rel_timestamp.microseconds)*1e-6 #convert to seconds
            data.append(np.array([rel_timestamp,float(line_att[2])])) #store the timestamp and current value of the sample

    return np.array(data) #convert the data list into a numpy array

def filter_integer_period(time_samples, n_periods, fm=60):
    '''
    :param time_samples: the time samples of the electrical signal
    :param n_periods: number of periods
    :param fm: fundamental frequency (60 Hz by default)
    :return: the index of the last sample of the n_periods
    '''
    time_disp = time_samples[0]+(n_periods/fm) #find the amount of time elapsed for the amount of periods
    if time_disp <= time_samples[-1]: #displacement must be smaller or equal to the existing time samples
        return np.argmin(np.abs(time_samples-time_disp)) #find the index of the last period sample
    else:
        raise ValueError(f'[filter_integer_period] n_periods={n_periods} is invalid for the signal')

class SimuData:
    '''
    class that stores all the required data from a finite element simulation from ANSYS
    '''
    def __init__(self, current_data, speed_data, load_percentage, ns, fm=60, n_periods=None, transient=False, normalize_by=np.max):
        '''
        :param current_data: the raw csv current output file from ANSYS in numpy format
        :param speed_data: the raw csv speed output file from ANSYS in numpy format
        :param load_percentage: percentage of the load used in the simulation [%]
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
        :param transient: flag to filter out the transient in the signal (False by default)
        :param normalize_by: which function will be used to normalize the FFT
        '''
        #check if current_data is a numpy array
        current_flag = True #flag to monitor which data to compute
        if current_data is None:
            print(f'[data_types] current_data returned None from file!')
            current_flag = False #set to False (do not compute current-related data)

        #check if current_data is a numpy array
        speed_flag = True #flag to monitor which data to compute
        if speed_data is None:
            print(f'[data_types] speed_data returned None from file!')
            speed_flag = False #set to False (do not compute speed-related data)

        #Input arguments
        self.current_data = current_data
        self.speed_data = speed_data
        self.load = load_percentage
        self.ns = ns
        self.fm = fm
        filter_periods = False #flag to apply integer period filtering or not
        if n_periods is not None:
            self.n_periods = int(n_periods)
            filter_periods = True #update the flag

        #define current and time samples from the raw data
        if current_flag:
            self.i_motor = self.current_data[:,1] #extract the phase current samples
            self.time_grid = self.current_data[:,0] #extract the time samples

            #extract the transient if required
            if not transient:
                transient_mask = (self.time_grid>=1) #mask the signal from 1s
                self.i_motor = self.i_motor[transient_mask]
                self.time_grid = self.time_grid[transient_mask]

            #extract the n integer periods if required
            if filter_periods:
                int_period_idx = filter_integer_period(self.time_grid, self.n_periods, fm=self.fm) #find the index of the last sample in the defined periods
                self.time_grid = self.time_grid[:int_period_idx+1]
                self.i_motor = self.i_motor[:int_period_idx+1]

            self.Ts = self.time_grid[1]-self.time_grid[0] #sampling time
            self.fs = 1/self.Ts #sampling frequency
            self.Res = self.fs/len(self.i_motor) #resolution

            #compute the spectrum of the current
            self.fft_freqs = np.linspace(-self.fs/2,self.fs/2,len(self.i_motor)) #FFT frequencies based on the sampling
            self.fft_data_amp = dsp_utils.compute_FFT(self.i_motor, normalize_by=normalize_by) #compute the FFT
            self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB

        #define some important electromagnetic variables
        if speed_flag:
            self.speed_motor = self.speed_data[:,1] #extract the speed samples
            self.speed_time_grid = self.speed_data[:,0] #extract the time samples

            #extract the transient if required
            if not transient:
                transient_mask = (self.speed_time_grid >= 1) #mask the signal from 1s
                self.speed_motor = self.speed_motor[transient_mask]
                self.speed_time_grid = self.speed_time_grid[transient_mask]

            #extract the n integer periods if required
            if filter_periods:
                int_period_idx = filter_integer_period(self.speed_time_grid, self.n_periods, fm=self.fm) #find the index of the last sample in the defined periods
                self.speed_motor = self.speed_motor[:int_period_idx+1]
                self.speed_time_grid = self.speed_time_grid[:int_period_idx+1]

            self.nr = electromag_utils.compute_avg_rotor_speed(self.speed_motor, self.speed_time_grid, self.fm) #compute the avg rotor speed
            self.slip = electromag_utils.compute_slip(self.ns, self.nr) #compute the slip

class PeakFinderData:
    def __init__(self, fft_data_dB, fft_freqs, fofd, smooth_data=None, slip=None, fm=None):
        self.fft_data_dB = fft_data_dB #store the fft spectrum in dB to avoid messing with the original data
        self.fft_freqs = fft_freqs #store the fft freqs to avoid messing with the oringal data
        self.fofd = fofd #store the gradient of the FFT (first order finite difference)
        self.smoothed = False #flag to monitor if the spectrumwas smoothed
        if smooth_data is not None:
            self.smoothed = True #update the smoothed flag
            self.smooth_fft_data_dB = smooth_data #store the smoothed fft spectrum in dB
        if slip is not None:
            self.slip = slip #update the slip of the simulated machine
        if fm is not None:
            self.fm = fm #update the fundamental frequency of the simulated machine

class LabData:
    def __init__(self, raw_data, fm=60, normalize_by=np.max):
        '''
        :param raw_data: the raw .mat output file from lab controlled tests in numpy format
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        '''
        #check if current_data is a numpy array
        if not isinstance(raw_data, np.ndarray):
            raise TypeError(f'[data_types] current_data input required to be a numpy array!')
        if len(raw_data) == 0:
            raise ValueError(f'[data_types] current_data passed as an empty array!')

        #Input arguments
        self.raw_data = raw_data
        self.fm = fm

        #define current and time samples from the raw data
        self.i_r = self.raw_data[:,1] #extract the R-phase current samples
        self.i_s = self.raw_data[:,2] #extract the S-phase current samples
        self.i_t = self.raw_data[:,3] #extract the T-phase current samples
        self.time_grid = self.raw_data[:,0] #extract the time samples
        self.Ts = self.time_grid[1]-self.time_grid[0] #sampling time
        self.fs = 1/self.Ts #sampling frequency
        self.Res = self.fs/len(self.i_r) #resolution

        #compute the spectrum of the currents
        self.fft_freqs = np.linspace(-self.fs/2, self.fs/2,len(self.i_r)) #FFT frequencies based on the sampling
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_t, normalize_by=normalize_by) #compute the FFT of the T phase
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
        self.fft_s_data_amp = dsp_utils.compute_FFT(self.i_s, normalize_by=normalize_by) #compute the FFT of the S phase
        self.fft_s_data_dB = dsp_utils.apply_dB(self.fft_s_data_amp) #convert from amplitude to dB
        self.fft_r_data_amp = dsp_utils.compute_FFT(self.i_r, normalize_by=normalize_by) #compute the FFT of the R phase
        self.fft_r_data_dB = dsp_utils.apply_dB(self.fft_r_data_amp) #convert from amplitude to dB

class SensorData:
    def __init__(self, raw_data, ns, fm=60, n_periods=None, transient=False, normalize_by=np.max):
        '''
        :param raw_data: the raw .MAT output file from lab controlled tests in numpy format
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
        :param transient: flag to filter out the transient
        :param normalize_by: which function will be used to normalize the FFT
        '''
        #check if current_data is a dictionary
        if not isinstance(raw_data, dict):
            raise TypeError(f'[data_types] current_data input required to be a dictionary!')
        if len(raw_data) == 0:
            raise ValueError(f'[data_types] current_data passed as an empty dictionary!')

        #Input arguments
        self.raw_data = raw_data
        self.fm = fm
        filter_periods = False #flag to apply integer period filtering or not
        if n_periods is not None:
            self.n_periods = int(n_periods)
            filter_periods = True #update the flag
        self.ns = ns

        #define current
        self.time_grid = self.raw_data["Channel_1_Data"] #extract the time samples

        #handle transient in the signal
        if transient:
            self.transient_mask = (self.time_grid>=0) #mask in order to keep the transient
        else:
            self.transient_mask = (self.time_grid>=8) #mask in order to filter out transient

        config_file = load_config_file('../config_file.yml') #load the config file
        self.time_grid = self.time_grid[self.transient_mask]
        self.i_r = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_5_Data"][self.transient_mask] #extract the R-phase current samples
        self.i_s = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_6_Data"][self.transient_mask] #extract the S-phase current samples
        self.i_t = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_7_Data"][self.transient_mask] #extract the T-phase current samples
        self.v_r = self.raw_data["Channel_2_Data"][self.transient_mask] #extract the R-phase voltage samples
        self.v_s = self.raw_data["Channel_3_Data"][self.transient_mask] #extract the S-phase voltage samples
        self.v_t = self.raw_data["Channel_4_Data"][self.transient_mask] #extract the T-phase voltage samples
        self.torque = self.raw_data["Channel_8_Data"][self.transient_mask] #extract the torque samples
        self.speed_motor = self.raw_data["Channel_9_Data"][self.transient_mask] #extract the speed samples

        #extract the n integer periods if required
        if filter_periods:
            int_period_idx = filter_integer_period(self.time_grid, self.n_periods, fm=self.fm) #find the index of the last sample in the defined periods
            self.time_grid = self.time_grid[:int_period_idx+1]
            self.i_r =  self.i_r[:int_period_idx+1]
            self.i_s = self.i_s[:int_period_idx+1]
            self.i_t = self.i_t[:int_period_idx+1]
            self.v_r = self.v_r[:int_period_idx+1]
            self.v_s = self.v_s[:int_period_idx+1]
            self.v_t = self.v_t[:int_period_idx+1]
            self.torque = self.torque[:int_period_idx+1]
            self.speed_motor = self.speed_motor[:int_period_idx+1]

        self.Ts = self.time_grid[1]-self.time_grid[0] #sampling time
        self.fs = 1/self.Ts #sampling frequency
        self.Res = self.fs/len(self.i_r) #resolution

        #compute the spectrum of the currents
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_t, normalize_by=normalize_by) #compute the FFT of the T phase
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
        self.fft_freqs = np.linspace(-self.fs/2, self.fs/2,len(self.i_r)) #FFT frequencies based on the sampling
        self.fft_s_data_amp = dsp_utils.compute_FFT(self.i_s, normalize_by=normalize_by) #compute the FFT of the S phase
        self.fft_s_data_dB = dsp_utils.apply_dB(self.fft_s_data_amp) #convert from amplitude to dB
        self.fft_r_data_amp = dsp_utils.compute_FFT(self.i_r, normalize_by=normalize_by) #compute the FFT of the R phase
        self.fft_r_data_dB = dsp_utils.apply_dB(self.fft_r_data_amp) #convert from amplitude to dB

        #define electromagnetic motor values
        self.speed_time_grid = self.time_grid #extract the time samples
        self.nr = electromag_utils.compute_avg_rotor_speed(self.speed_motor, self.speed_time_grid, self.fm) #compute the avg rotor speed
        self.slip = electromag_utils.compute_slip(self.ns, self.nr) #compute the slip

class BatchSensorData:
    def __init__(self, filelist, load_percentage, ns, fm=60, n_periods=None, transient=False, normalize_by=np.max):
        '''
        :param filelist: list with the .MAT output file in the local filesystem
        :param load_percentage: percentage of the load used in the simulation [%]
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
        :param transient: flag to filter out the transient
        :param normalize_by: which function will be used to normalize the FFT
        '''
        #Input arguments
        self.batch_list = filelist #list with the batch of data
        self.load_percentage = load_percentage #load percentage to append data
        self.ns = ns
        self.fm = fm
        self.transient = transient

        #Compute the average of the current and voltage in every phase
        self.batch_data = np.zeros_like(self.batch_list, dtype=SensorData) #list to append all the SensorData structures computed per file
        for batch_idx in range(len(self.batch_list)):
            batch_file = self.batch_list[batch_idx] #load the current file based on the index

            #check if file is valid
            if not os.path.isfile(batch_file):
                raise ValueError(f'[BatchSensorData] File {batch_file} does not exist!')

            curr_raw_data = loadmat(batch_file) #read the raw .MAT sensor_file into dictionary format
            self.batch_data[batch_idx] = SensorData(curr_raw_data, self.ns, fm=self.fm, n_periods=n_periods, transient=self.transient,
                                                    normalize_by=normalize_by) #append the data structure with the lab tested output
            print(f'[BatchSensorData] File {batch_file} read!')

class NIHardwareData:
    def __init__(self, file, fm=60, n_periods=None, normalize_by=np.max):
        '''
        :param file: path to the .txt output file in the local filesystem
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
        :param normalize_by: which function will be used to normalize the FFT
        '''
        #Input arguments
        if not os.path.isfile(file):
            raise ValueError(f'[NIHardwareData] File {file} does not exist!')

        self.file = file #path to the
        self.fm = fm
        filter_periods = False #flag to apply integer period filtering or not
        if n_periods is not None:
            self.n_periods = int(n_periods)
            filter_periods = True #update the flag
        self.raw_data = read_NIHW_data(file) #read the .txt file content into a numpy array
        self.time_grid = self.raw_data[:,0] #time samples
        self.i_motor = self.raw_data[:,1] #current samples

        #extract the n integer periods if required
        if filter_periods:
            int_period_idx = filter_integer_period(self.time_grid, self.n_periods, fm=self.fm) #find the index of the last sample in the defined periods
            self.time_grid = self.time_grid[:int_period_idx+1]
            self.i_motor = self.i_motor[:int_period_idx+1]

        self.Ts = self.time_grid[1]-self.time_grid[0] #sampling time
        self.fs = 1/self.Ts #sampling frequency
        self.Res = self.fs/len(self.i_motor) #resolution

        #compute the spectrum of the current
        self.fft_freqs = np.linspace(-self.fs/2, self.fs/2, len(self.i_motor)) #FFT frequencies based on the sampling
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_motor, normalize_by=normalize_by) #compute the FFT
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB

class BatchNIHardwareData:
    def __init__(self, file_list, fm=60, n_periods=None, normalize_by=np.max):
        '''
        :param file_list: list containing all the files to append in the batch
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param n_periods: the integer number of periods that will be extracted from the current data (None by default=all samples)
        :param normalize_by: which function will be used to normalize the FFT
        '''
        self.file_list = file_list
        self.fm = fm
        self.batch_data = np.zeros_like(self.file_list, dtype=NIHardwareData) #list to append all the NIHardwareData structures computed per file
        for file_idx in range(len(file_list)):
            self.batch_data[file_idx] = NIHardwareData(file_list[file_idx], fm=self.fm, n_periods=n_periods, normalize_by=normalize_by) #append the data structure with the hardware output
            print(f'[BatchNIHardwareData] File {file_list[file_idx]} read!')
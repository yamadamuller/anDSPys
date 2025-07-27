import os
import numpy as np
from framework import electromag_utils, dsp_utils
from scipy.io import loadmat
import yaml

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

class SimuData:
    '''
    class that stores all the required data from a finite element simulation from ANSYS
    '''
    def __init__(self, current_data, speed_data, load_percentage, ns, fm=60):
        '''
        :param current_data: the raw csv current output file from ANSYS in numpy format
        :param speed_data: the raw csv speed output file from ANSYS in numpy format
        :param load_percentage: percentage of the load used in the simulation [%]
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
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

        #define current and time samples from the raw data
        if current_flag:
            self.i_motor = self.current_data[:,1] #extract the phase current samples
            self.i_time_grid = self.current_data[:,0] #extract the time samples
            self.Ts = self.i_time_grid[1]-self.i_time_grid[0] #sampling time
            self.fs = 1/self.Ts #sampling frequency
            self.Res = self.fs/len(self.i_motor) #resolution

            #compute the spectrum of the current
            self.fft_data_amp = dsp_utils.compute_FFT(self.i_motor) #compute the FFT
            self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
            self.fft_freqs = np.linspace(-self.fs/2, self.fs/2, len(self.i_motor)) #FFT frequencies based on the sampling

        #define some important electromagnetic variables
        if speed_flag:
            self.speed_motor = self.speed_data[:,1] #extract the speed samples
            self.speed_time_grid = self.speed_data[:,0] #extract the time samples
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
    def __init__(self, raw_data, fm=60):
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
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_t) #compute the FFT of the T phase
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
        self.fft_freqs = np.linspace(-self.fs/2, self.fs/2, len(self.i_r)) #FFT frequencies based on the sampling
        self.fft_s_data_amp = dsp_utils.compute_FFT(self.i_s) #compute the FFT of the S phase
        self.fft_s_data_dB = dsp_utils.apply_dB(self.fft_s_data_amp) #convert from amplitude to dB
        self.fft_r_data_amp = dsp_utils.compute_FFT(self.i_r) #compute the FFT of the R phase
        self.fft_r_data_dB = dsp_utils.apply_dB(self.fft_r_data_amp) #convert from amplitude to dB

class SensorData:
    def __init__(self, raw_data, ns, fm=60, transient=False):
        '''
        :param raw_data: the raw .MAT output file from lab controlled tests in numpy format
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param transient: flag to filter out the transient
        '''
        #check if current_data is a dictionary
        if not isinstance(raw_data, dict):
            raise TypeError(f'[data_types] current_data input required to be a dictionary!')
        if len(raw_data) == 0:
            raise ValueError(f'[data_types] current_data passed as an empty dictionary!')

        #Input arguments
        self.raw_data = raw_data
        self.fm = fm
        self.ns = ns

        #define current
        self.time_grid = self.raw_data["Channel_1_Data"] #extract the time samples
        #TODO: find better ways to filter transient
        #self.time_grid[-1]-10
        if transient:
            self.transient_mask = (self.time_grid>=0) #mask in order to keep the transient
        else:
            self.transient_mask = (self.time_grid>=7.5) #mask in order to filter out transient

        config_file = load_config_file('../config_file.yml') #load the config file
        self.time_grid = self.time_grid[self.transient_mask]
        self.i_r = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_5_Data"][self.transient_mask] #extract the R-phase current samples
        self.i_s = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_6_Data"][self.transient_mask] #extract the S-phase current samples
        self.i_t = int(config_file["sensor-configs"]["current_relation"])*self.raw_data["Channel_7_Data"][self.transient_mask] #extract the T-phase current samples
        self.Ts = self.time_grid[1]-self.time_grid[0] #sampling time
        self.fs = 1/self.Ts #sampling frequency
        self.Res = self.fs/len(self.i_r) #resolution

        #compute the spectrum of the currents
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_t) #compute the FFT of the T phase
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
        self.fft_freqs = np.linspace(-self.fs/2, self.fs/2,len(self.i_r)) #FFT frequencies based on the sampling
        self.fft_s_data_amp = dsp_utils.compute_FFT(self.i_s) #compute the FFT of the S phase
        self.fft_s_data_dB = dsp_utils.apply_dB(self.fft_s_data_amp) #convert from amplitude to dB
        self.fft_r_data_amp = dsp_utils.compute_FFT(self.i_r) #compute the FFT of the R phase
        self.fft_r_data_dB = dsp_utils.apply_dB(self.fft_r_data_amp) #convert from amplitude to dB

        #define other motor values
        self.v_r = self.raw_data["Channel_2_Data"][self.transient_mask] #extract the R-phase voltage samples
        self.v_s = self.raw_data["Channel_3_Data"][self.transient_mask] #extract the S-phase voltage samples
        self.v_t = self.raw_data["Channel_4_Data"][self.transient_mask] #extract the T-phase voltage samples
        self.torque = self.raw_data["Channel_8_Data"][self.transient_mask] #extract the torque samples
        self.speed_motor = self.raw_data["Channel_9_Data"][self.transient_mask] #extract the speed samples
        self.speed_time_grid = self.time_grid #extract the time samples
        self.nr = electromag_utils.compute_avg_rotor_speed(self.speed_motor, self.speed_time_grid, self.fm) #compute the avg rotor speed
        self.slip = electromag_utils.compute_slip(self.ns, self.nr) #compute the slip

class BatchSensorData:
    def __init__(self, filedir, load_percentage, ns, fm=60, transient=False):
        '''
        :param filedir: path to the .csv output file in the local filesystem
        :param load_percentage: percentage of the load used in the simulation [%]
        :param ns: synchronous speed of the simulated motor [rpm]
        :param fm: fundamental frequency [Hz] (60 Hz by default)
        :param transient: flag to filter out the transient
        '''
        #Input arguments
        self.filedir = filedir #directory with the batch of data
        self.load_percentage = load_percentage #load percentage to append data
        self.batch_list = os.listdir(self.filedir) #list all the directory files
        self.batch_list = [os.path.join(filedir, batch_file) for batch_file in self.batch_list if str(load_percentage) in batch_file] #filter all the files with the same load percentage
        self.batch_list.sort() #sort the list
        self.ns = ns
        self.fm = fm
        self.transient = transient

        #Compute the average of the current and voltage in every phase
        self.sdata_list = [] #list to append all the SensorData structures computed per file
        for sensor_file in self.batch_list:
            curr_raw_data = loadmat(sensor_file) #read the raw .MAT sensor_file into dictionary format
            self.sdata_list.append(SensorData(curr_raw_data, self.ns, self.fm, self.transient)) #append the data structure with the lab tested output
            print(f'[BatchSensorData] File {sensor_file} read!')
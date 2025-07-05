import numpy as np
from framework import electromag_utils, dsp_utils

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
        if not isinstance(current_data, np.ndarray):
            raise TypeError(f'[data_types] current_data input required to be a numpy array!')
        if len(current_data)==0:
            raise ValueError(f'[data_types] current_data passed as an empty array!')

        #check if current_data is a numpy array
        if not isinstance(speed_data, np.ndarray):
            raise TypeError(f'[data_types] speed_data input required to be a numpy array!')
        if len(speed_data)==0:
            raise ValueError(f'[data_types] speed_data passed as an empty array!')

        #Input arguments
        self.current_data = current_data
        self.speed_data = speed_data
        self.load = load_percentage
        self.ns = ns
        self.fm = fm

        #define current and time samples from the raw data
        self.i_motor = self.current_data[:,1] #extract the phase current samples
        self.i_time_grid = self.current_data[:,0] #extract the time samples
        self.speed_motor = self.speed_data[:,1] #extract the speed samples
        self.speed_time_grid = self.speed_data[:,0] #extract the time samples
        self.Ts = self.i_time_grid[1]-self.i_time_grid[0] #sampling time
        self.fs = 1/self.Ts #sampling frequency
        self.Res = self.fs/len(self.i_motor) #resolution

        #define some important electromagnetic variables
        self.nr = electromag_utils.compute_avg_rotor_speed(self.speed_motor, self.speed_time_grid, self.fm) #compute the avg rotor speed
        self.slip = electromag_utils.compute_slip(self.speed_motor, self.speed_time_grid, self.ns, self.nr) #compute the slip

        #compute the spectrum of the current
        self.fft_data_amp = dsp_utils.compute_FFT(self.i_motor) #compute the FFT
        self.fft_data_dB = dsp_utils.apply_dB(self.fft_data_amp) #convert from amplitude to dB
        self.fft_freqs = np.linspace(-self.fs / 2, self.fs / 2, len(self.i_motor)) #FFT frequencies based on the sampling

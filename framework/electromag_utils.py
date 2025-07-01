import numpy as np

def compute_avg_rotor_speed(speed_data, time_samples, fm=60):
    '''
    :param speed_data: speed magnitude data in array format [rpm]
    :param time_samples: time samples from the simulation in array format [s]
    :param fm: fundamental frequency [Hz] (60Hz by default)
    :return: the average for the simulated velocities in the last period
    '''
    #check if speed_data is a numpy array
    if not isinstance(speed_data, np.ndarray):
        raise TypeError(f'[compute_avg_rotor_speed] speed_data input required to be a numpy array!')
    if len(speed_data)==0:
        raise ValueError(f'[compute_avg_rotor_speed] speed_data passed as an empty array!')

    #check if time_samples is a numpy array
    if not isinstance(time_samples, np.ndarray):
        raise TypeError(f'[compute_avg_rotor_speed] time_samples input required to be a numpy array!')
    if len(time_samples)==0:
        raise ValueError(f'[compute_avg_rotor_speed] time_samples passed as an empty array!')

    lower_lim = time_samples[-1] - (1/fm) #last_sample - 1/60, last period from the samples to compute the avg. speed
    time_mask = time_samples>=lower_lim #mask to filter only the last period from the speed data
    return np.mean(speed_data[time_mask])

def compute_slip(speed_data, time_samples, ns, nr):
    '''
    :param speed_data: speed magnitude data in array format [rpm]
    :param time_samples: time samples from the simulation in array format [s]
    :param ns: synchronous speed of the simulated motor [rpm]
    :param nr: the rotor speed of the simulated motor [rpm]
    :return: the slip of the simulated motor
    '''
    #check if speed_data is a numpy array
    if not isinstance(speed_data, np.ndarray):
        raise TypeError(f'[compute_slip] speed_data input required to be a numpy array!')
    if len(speed_data)==0:
        raise ValueError(f'[compute_slip] speed_data passed as an empty array!')

    #check if time_samples is a numpy array
    if not isinstance(time_samples, np.ndarray):
        raise TypeError(f'[compute_slip] time_samples input required to be a numpy array!')
    if len(time_samples)==0:
        raise ValueError(f'[compute_slip] time_samples passed as an empty array!')

    return (ns-nr)/ns
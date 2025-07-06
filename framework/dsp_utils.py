import numpy as np
from framework import data_types, dsp_utils

def apply_autocontrast(signal):
    '''
    :param signal: the array with the signal to be normalized
    :return: the signal normalized
    '''
    return (signal - np.min(signal))/np.abs(np.max(signal) - np.min(signal))

def apply_dB(signal):
    '''
    :param signal: the array with the signal to be converted from amplitude to dB
    :return: 20*log10(abs(signal))
    '''
    return 20*np.log10(np.abs(signal)+1e-8)

def apply_IQM(signal):
    '''
    :param signal: the array with the signal to be computed both upper and lower limits of the IQR
    :return: Q1-(1.5*IQR), Q3+(1.5*IQR)
    '''
    Q1 = np.percentile(signal, 25) #first quantile (below 25%)
    Q3 = np.percentile(signal, 75) #third quantile (above 75%)
    IQR = Q3-Q1 #inter-quantile range
    lower = Q1-(1.5*IQR) #lower limit
    upper = Q3+(1.5*IQR) #upper limit
    return [lower, upper]

def apply_stat_filt(signal, filter_function, abs=False):
    '''
    :param signal: the array with the signal to be filtered
    :param filter_function: which statitical function will be used to filter out values
    :param abs: flag to filter based on the absolute values of the signal
    :return: the filtered array mask (values >= limit)
    '''
    stat_threshold = filter_function(signal) #statistcial component

    if abs:
        signal = np.abs(signal) #extract the absolute values

    if type(stat_threshold) == list:
        return(signal<=stat_threshold[0])&(signal<=stat_threshold[1])
    else:
        return signal>=stat_threshold

def compute_FFT(signal, shift=True, normalize=True):
    '''
    :param signal: the array with the signal to be transformed into the frequency domain
    :param shift: flag to shift or not the FFT output (True by default)
    :param normalize: flag to return the FFT magnitude normalized or not (True by default)
    :return: spectrum of the signal
    '''
    fft = np.fft.fft(signal) #compute the fft on the signal

    if shift:
        fft = np.fft.fftshift(fft) #center in 0 Hz

    if normalize:
        fft = np.abs(fft)/np.max(fft) #normalized by the maximum value

    return fft

def compute_DFT(signal_r, signal_i, shift=True):
    '''
    :param signal_r: real part of the signal
    :param signal_i: complex part of the signal
    :param shift: flag to shift or not the FFT output (True by default)
    :return: real and complex parts of the spectrum
    '''
    N = len(signal_r) #total samples

    #force 1xN size
    signal_r = np.atleast_2d(signal_r)
    signal_i = np.atleast_2d(signal_i)

    if shift:
        u = np.arange(0, N, 1) #frequency components
        signal_r = signal_r * (-1)**u #shift real
        signal_i = signal_i * (-1)**u  #shift complex

    #Vectorized DFT
    n = np.arange(0, N, 1) #0:N-1 interval
    kx, ky = np.meshgrid(n, n) #2D matrix of the 0:N-1 interval
    theta = 2 * np.pi * ky * n / N #theta = (2*pi*u/N)*(n-1)
    dft_r = signal_r @ np.cos(theta) + 1j * signal_i @ np.sin(theta) #r[n]*cos(theta) + fi[n]*sen(theta)
    dft_i = 1j * signal_i @ np.cos(theta) - signal_r @ np.sin(theta) #j{-fr[n]*sen(theta) + fi[n]*cos(theta)}

    return dft_r.ravel(), dft_i.ravel()

def central_num_spectrum_gradient(spectrum, eps=1):
    '''
    :param spectrum: spectrum of the signal in array format
    :param eps: epsilon (smallest change possible)
    :return: numerical gradient computed by central difference
    '''
    #check if speed_data is a numpy array
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(f'[num_spectrum_gradient] spectrum input required to be a numpy array!')
    if len(spectrum) == 0:
        raise ValueError(f'[num_spectrum_gradient] spectrum passed as an empty array!')

    backward = np.roll(spectrum, eps) #lagged signal x[i-1]
    forward = np.roll(spectrum, -eps) #advanced signal x[i+1]
    return (forward - backward)/(2*eps) #central finite difference (x[i+1] - x[i-1])/2eps

def backwards_num_spectrum_gradient(spectrum, eps=1):
    '''
    :param spectrum: spectrum of the signal in array format
    :param eps: epsilon (smallest change possible)
    :return: numerical gradient computed by central difference
    '''
    #check if speed_data is a numpy array
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(f'[num_spectrum_gradient] spectrum input required to be a numpy array!')
    if len(spectrum) == 0:
        raise ValueError(f'[num_spectrum_gradient] spectrum passed as an empty array!')

    backward = np.roll(spectrum, eps) #lagged signal x[i-1]
    return (spectrum - backward)/eps #backwards finite difference (x[i+1] - x[i-1])/2eps

def central_num_spectrum_hessian(spectrum, eps=1):
    '''
    :param spectrum: spectrum of the signal in array format
    :param eps: epsilon (smallest change possible)
    :return: numerical gradient computed by central difference
    '''
    #check if speed_data is a numpy array
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(f'[num_spectrum_gradient] spectrum input required to be a numpy array!')
    if len(spectrum) == 0:
        raise ValueError(f'[num_spectrum_gradient] spectrum passed as an empty array!')

    backward = np.roll(spectrum, eps) #lagged signal x[i-1]
    forward = np.roll(spectrum, -eps) #advanced signal x[i+1]
    return (forward - 2*spectrum + backward)/(eps**2) #central finite difference (x[i+1] - 2*x[i] +x[i-1])/eps²

def backwards_num_spectrum_hessian(spectrum, eps=1):
    '''
    :param spectrum: spectrum of the signal in array format
    :param eps: epsilon (smallest change possible)
    :return: numerical gradient computed by central difference
    '''
    #check if speed_data is a numpy array
    if not isinstance(spectrum, np.ndarray):
        raise TypeError(f'[num_spectrum_gradient] spectrum input required to be a numpy array!')
    if len(spectrum) == 0:
        raise ValueError(f'[num_spectrum_gradient] spectrum passed as an empty array!')

    backward = np.roll(spectrum, eps) #lagged signal x[i-1]
    backwards2 = np.roll(spectrum, 2*eps) #lagged signal x[i-2]
    return (spectrum - 2*backward + backwards2)/(eps**2) #central finite difference (x[i+1] - 2*x[i] +x[i-1])/eps²

def find_peaks(data, grad, hess, harm_component, harm_idx, bound_idx, window_limit=None, mag_threshold=None, freq_threshold=None, max_peaks=None, stat_filt=False, stat_func=np.mean):
    '''
    :param data: framework.data_types.SimuData structure
    :param grad: first order component of the FFT
    :param hess: second order component of the FFT
    :param harm_component: the harmonic component
    :param harm_idx: the index of the peak of the FFT magnitude
    :param bound_idx: the index of the boundary of the window (lower or upper)
    :param window_limit: index to reduce the window on each side of the global maximum (None by default=0)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-80dB)
    :param freq_threshold: threshold for frequency values below/above n+-threshold
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param stat_filt: flag to filter smaller values in the FFT than the given stat_func output
    :param stat_func: which function to compute in order to filter out small values (mean by default)
    :return: the coordinates for the six most significant sideband peaks of the computed harmonic component
    output[n] = coordinates -> [freqs, magnitudes]
    '''
    if type(data) != data_types.SimuData:
        raise TypeError(f'[find_peaks] data input must be a SimuData object!')
    if harm_idx == bound_idx:
        raise ValueError(f'[find_peaks] harm_idx and bound_idx must not be the same!')
    if not window_limit:
        window_limit = 0 #set the index to zero, compute over all points
    if not mag_threshold:
        mag_threshold = -80 #find where the FFT lies above -80 dB
    if not freq_threshold:
        freq_threshold = 0 #detect any peak as close as possible from the harmonic spike
    if not stat_func:
        stat_func = np.mean #in case stat_func is passed as None

    #Find the peaks within the side defined by the boundary index (bound_idx)
    if bound_idx < harm_idx: #compute the peaks left of the component
        wind_freqs = data.fft_freqs[bound_idx:harm_idx-int(window_limit)]
        wind_spectrum = data.fft_data_dB[bound_idx:harm_idx-int(window_limit)]
        wind_grad = grad[bound_idx:harm_idx-int(window_limit)]
        wind_hess = hess[bound_idx:harm_idx-int(window_limit)]
        freq_thresh_mask = wind_freqs<=(harm_component-freq_threshold) #avoid detecting peaks close to the component spike
    else:
        wind_freqs = data.fft_freqs[harm_idx+int(window_limit):bound_idx]
        wind_spectrum = data.fft_data_dB[harm_idx+int(window_limit):bound_idx]
        wind_grad = grad[harm_idx+int(window_limit):bound_idx]
        wind_hess = hess[harm_idx+int(window_limit):bound_idx]
        freq_thresh_mask = wind_freqs>=(harm_component+freq_threshold) #avoid detecting peaks close to the component spike

    #Evaluate signal change in the first derivative to infer local maxima
    wind_grad[~apply_stat_filt(wind_grad, np.mean, abs=True)] = 0 #filter out small gradient values that might indicate significant peaks close to the spike
    grad_sign = np.sign(wind_grad) #compute the signs of each value of the first derivative
    grad_sign_change = np.roll(grad_sign,-1)+grad_sign #signs[i+1]-signs[i]

    #Extract the peaks
    sign_change_mask = grad_sign_change == 0 #when the first derivative changes from + to -. the sum is 0
    mag_thresh_mask = wind_spectrum >= mag_threshold #find where the FFT lies above a pre-defined threshold
    peaks_mask = sign_change_mask&mag_thresh_mask&freq_thresh_mask #logic operation to define where all three masks are valid
    peaks = np.stack((wind_freqs[peaks_mask], wind_spectrum[peaks_mask]),axis=1) #stack the peaks as [freqs, coordinates]

    if stat_filt:
        #filter out small peak values that might indicate significant peaks close to the spike
        stat_mask = apply_stat_filt(peaks[:,0], stat_func)
        peaks = peaks[stat_mask,:]

    if max_peaks:
        if len(peaks) >= max_peaks:
            return peaks[-max_peaks:]
        else:
            raise ValueError (f'[find_peaks] length of the peaks matrix is smaller than max_peaks!')
    else:
        return peaks


def fft_significant_peaks(data, harm_components, window_size=None, window_limit=None, mag_threshold=None, freq_threshold=None, max_peaks=None, stat_filt=False, stat_func=np.mean):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_components: a list/array containing the harmonic components to iterate over
    :param window_size: the size in Hz of the window around each harmonic component (None by default=50Hz)
    :param window_limit: index to reduce the window on each side of the global maximum (None by default=0)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-80dB)
    :param freq_threshold: threshold for frequency values below/above n+-threshold
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param stat_filt: flag to filter smaller values in the FFT than the given stat_func output
    :param stat_func: which function to compute in order to filter out small values (mean by default)
    :return: the coordinates for the six most significant sideband peaks per harmonic component
    len(output) = 3
    output[n] = coordinates
    len(coordinates) <= 6 -> [freq, magnitude]
    '''
    if type(data) != data_types.SimuData:
        raise TypeError(f'[fft_peak_finder] data input must be a SimuData object!')
    if not window_size:
        window_size = 50 #window of 50 Hz around the harmonic spike
    if not window_limit:
        window_limit = 0 #set the index to zero, compute over all points
    if not mag_threshold:
        mag_threshold = -80 #find where the FFT lies above -80 dB
    if not freq_threshold:
        freq_threshold = 0 #detect any peak as close as possible from the harmonic spike
    if not stat_func:
        stat_func = np.mean #in case stat_func is passed as None

    #Filter only positive values from the fft frequencies
    freq_mask = data.fft_freqs>=0 #mask to filter negative frequencies
    data.fft_freqs = data.fft_freqs[freq_mask] #filtered frequencies
    data.fft_data_amp = data.fft_data_amp[freq_mask] #filtered magnitudes amplitude
    data.fft_data_dB = data.fft_data_dB[freq_mask] #filtered magnitudes dB

    #Compute the first and second derivatives of the spectrum
    fofd = dsp_utils.backwards_num_spectrum_gradient(data.fft_data_dB, eps=1) #backwards first order finite difference
    sofd = dsp_utils.backwards_num_spectrum_hessian(data.fft_data_dB, eps=1) #backwards second order finite difference

    #iterate over the harmonic components to find all the significant sideband peaks
    peaks_per_component = [] #list to append the detected peaks for each harmonic component
    for n in harm_components:
        #Window the FFT signal around the peak of the harmonic component
        n = data.fm*n #update the component as a ratio of the fundamental frequency
        lower_idx = np.argmin(np.abs(data.fft_freqs-(n-int(window_size/2)))) #window limit on the left
        upper_idx = np.argmin(np.abs(data.fft_freqs-(n+int(window_size/2)))) #window limit on the right
        harm_idx = np.argmin(np.abs(data.fft_freqs-n)) #find the index of the harmonic component peak

        #extract the peaks for both window sides
        lower_peaks = find_peaks(data, fofd, sofd, n, harm_idx, lower_idx, window_limit=window_limit, mag_threshold=mag_threshold,
                                 freq_threshold=freq_threshold, max_peaks=max_peaks, stat_filt=stat_filt, stat_func=stat_func) #left side of the spike
        upper_peaks = np.zeros_like(lower_peaks) #TODO: compute the peaks at the right side of the spike
        peaks_per_component.append(np.concat([lower_peaks, upper_peaks])) #concatenate the peaks

    return peaks_per_component

def fft_IQM_per_harmonic(data, harm_components, spike_side=None, window_size=None, window_limit=None):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_components: a list/array containing the harmonic components to iterate over
    :param spike_side: which side of the harmonic spike to compute the IQM (None by default=left)
    :param window_size: the size in Hz of the window around each harmonic component (None by default=50Hz)
    :param window_limit: index to reduce the window on each side of the global maximum (None by default=0)
    :return: the coordinates for the six most significant sideband peaks per harmonic component
    len(output) = 3
    output[n] = coordinates
    len(coordinates) <= 6 -> [freq, magnitude]
    '''
    if type(data) != data_types.SimuData:
        raise TypeError(f'[fft_peak_finder] data input must be a SimuData object!')
    if not spike_side:
        spike_side='left'
    if(spike_side!='left')&(spike_side!='right'):
        raise ValueError(f'[fft_IQM_per_harmonic] spike_side must be "left" or "right"!')
    if not window_size:
        window_size = 50 #window of 50 Hz around the harmonic spike
    if not window_limit:
        window_limit = 0 #set the index to zero, compute over all points

    #Filter only positive values from the fft frequencies
    freq_mask = data.fft_freqs>=0 #mask to filter negative frequencies
    data.fft_freqs = data.fft_freqs[freq_mask] #filtered frequencies
    data.fft_data_amp = data.fft_data_amp[freq_mask] #filtered magnitudes amplitude
    data.fft_data_dB = data.fft_data_dB[freq_mask] #filtered magnitudes dB

    #iterate over the harmonic components to find all the significant sideband peaks
    iqm_per_component = [] #list to append the detected peaks for each harmonic component
    for n in harm_components:
        #Window the FFT signal around the peak of the harmonic component
        n = data.fm*n #update the component as a ratio of the fundamental frequency
        lower_idx = np.argmin(np.abs(data.fft_freqs-(n-int(window_size/2)))) #window limit on the left
        upper_idx = np.argmin(np.abs(data.fft_freqs-(n+int(window_size/2)))) #window limit on the right
        harm_idx = np.argmin(np.abs(data.fft_freqs-n)) #find the index of the harmonic component peak

        # Find the peaks within the side defined by the boundary index (bound_idx)
        if spike_side=='left':  # compute the peaks left of the component
            wind_spectrum = data.fft_data_dB[lower_idx:harm_idx - int(window_limit)]
        else:
            wind_spectrum = data.fft_data_dB[harm_idx + int(window_limit):upper_idx]

        iqm_per_component.append(apply_IQM(wind_spectrum)) #compute the IQM boundaries for the curve

    return iqm_per_component
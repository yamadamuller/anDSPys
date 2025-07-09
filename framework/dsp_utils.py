import numpy as np
from framework import data_types, dsp_utils
from scipy.signal import savgol_filter

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

def apply_moving_filter(signal, stat_func, window_size):
    '''
    :param signal: the array with the signal to apply a moving filter operation
    :param stat_func: the function that will be applied to the slidding window
    :param window_size: size of the sliding window
    :return: the signal filtered by the sliding window
    '''
    filtered_signal = np.zeros_like(signal)
    signal = np.pad(signal, (window_size-1, 0)) #pad with zeros so every element in signal is computed
    for sldwin in range(window_size-1, len(signal)):
        filtered_signal[sldwin-window_size-1] = stat_func(signal[sldwin-window_size-1:sldwin+1]) #compute the moving filter

    return filtered_signal

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

def fft_significant_peaks(data, harm_components, window_size=None, window_limit=None, mag_thresholds=None, freq_threshold=None, max_peaks=None, stat_filt=False, stat_func=np.mean):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_components: a list/array containing the harmonic components to iterate over
    :param window_size: the size in Hz of the window around each harmonic component (None by default=50Hz)
    :param window_limit: index to reduce the window on each side of the global maximum (None by default=0)
    :param mag_thresholds: thresholds for magnitude values of the FFT [dB] at each harmonic component (None by default=-80dB)
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
    if not mag_thresholds:
        mag_thresholds = [-80, -80, -80] #find where the FFT lies above -80 dB
    if len(mag_thresholds)!=len(harm_components):
        raise ValueError(f'[fft_peak_finder] mag_thresholds and harm_components must have the same size!')
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
    local_thresholds = mag_thresholds.copy() #copy the threshold list
    local_thresholds.reverse() #reverse the list to pop each element at a time
    for n in harm_components:
        #Window the FFT signal around the peak of the harmonic component
        n = data.fm*n #update the component as a ratio of the fundamental frequency
        lower_idx = np.argmin(np.abs(data.fft_freqs-(n-int(window_size/2)))) #window limit on the left
        #upper_idx = np.argmin(np.abs(data.fft_freqs-(n+int(window_size/2)))) #window limit on the right
        harm_idx = np.argmin(np.abs(data.fft_freqs-n)) #find the index of the harmonic component peak

        #extract the peaks for both window sides
        lower_peaks = find_peaks(data, fofd, sofd, n, harm_idx, lower_idx, window_limit=window_limit, mag_threshold=local_thresholds.pop(),
                                 freq_threshold=freq_threshold, max_peaks=max_peaks, stat_filt=stat_filt, stat_func=stat_func) #left side of the spike
        #upper_peaks = np.zeros_like(lower_peaks) #TODO: compute the peaks at the right side of the spike
        peaks_per_component.append(np.concat([lower_peaks, np.array([[data.fft_freqs[harm_idx],data.fft_data_dB[harm_idx]]])])) #concatenate the peaks

    return peaks_per_component

def dispersion_find_peaks(data, lower_bound_idx, upper_bound_idx, kernel_size=None, gamma=None, mag_threshold=None, max_peaks=None, harm_component=None):
    '''
    :param data: framework.data_types.PeakFinderData structure
    :param lower_bound_idx: the index of the boundary of the window (lower side)
    :param upper_bound_idx: the index of the boundary of the window (upper side)
    :param kernel_size: size of the moving average and std filter kernel (None by default=5)
    :param gamma: threshold for the difference between a peak and the moving average (None by default=1)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-80dB)
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param harm_component: the harmonic component to filter max peaks from (only used if max_peaks is not None)
    :return: the coordinates for the six most significant sideband peaks of the computed harmonic component
    output[n] = coordinates -> [freqs, magnitudes]
    '''
    if type(data) != data_types.PeakFinderData:
        raise TypeError(f'[dispersion_find_peaks] data input must be a PeakFinderData object!')
    if lower_bound_idx == upper_bound_idx:
        raise ValueError(f'[dispersion_find_peaks] lower and upper boundary of the signal window must not be the same!')
    if not kernel_size:
        kernel_size = 5 #set the moving filter kernel to 5x5
    if not gamma:
        gamma = 1 #set the difference threshold as two time the moving standart deviation
    if not mag_threshold:
        mag_threshold = -60 #set the magnitude threshold as -60dB

    #Find the peaks within the side defined by the boundary index (bound_idx)
    wind_freqs = data.fft_freqs[lower_bound_idx:upper_bound_idx]
    wind_spectrum = data.fft_data_dB[lower_bound_idx:upper_bound_idx]
    wind_grad = data.fofd[lower_bound_idx:upper_bound_idx] #gradient of the spectrum (first order finite difference)

    #Compute the moving filters
    mov_avg = np.convolve(wind_spectrum, np.ones(kernel_size)/kernel_size, mode='same') #moving average
    mov_std = apply_moving_filter(wind_spectrum,np.std,kernel_size) #moving standard deviation

    #Evaluate signal change in the first derivative to infer local maxima
    grad_sign = np.sign(wind_grad) #compute the signs of each value of the first derivative
    grad_sign_change = np.roll(grad_sign,-1)+grad_sign #signs[i+1]-signs[i]

    #Evaluate spectrum peaks based on its neighbourhood
    wind_l_upeaks = wind_spectrum>np.roll(wind_spectrum, 1) #check if a sample up peak is greater than its left neighbour
    wind_l_dpeaks = np.abs(wind_spectrum)>np.abs(np.roll(wind_spectrum, 1)) #check if a sample down peak is greater than its left neighbour
    wind_r_upeaks = wind_spectrum>np.roll(wind_spectrum, -1) #check if a sample up peak is greater its right neighbour
    wind_r_dpeaks = wind_spectrum>np.roll(wind_spectrum, -1) #check if a sample down peaks is greater its right neighbour
    neighbour_mask = (wind_l_upeaks|wind_l_dpeaks)&(wind_r_upeaks|wind_r_dpeaks) #check for samples where the values are greater than its neighbour (both absolute and )

    #Extract the peaks
    sign_change_mask = grad_sign_change == 0 #when the first derivative changes from + to -. the sum is 0
    mag_thresh_mask = wind_spectrum>=mag_threshold #values of the FFT that surpass the magnitude threshold
    sign_change_mask = neighbour_mask&sign_change_mask&mag_thresh_mask #update the mask where all prior masks are valid
    raw_peaks = wind_spectrum[sign_change_mask] #every peak magnitude detected by the change of signal in the gradient
    raw_freq_peaks = wind_freqs[sign_change_mask] #every peak frequency detected by the change of signal in the gradient
    mov_dif = np.abs(raw_peaks-mov_avg[sign_change_mask]) #difference between a sample and the moving mean at its index
    stat_peaks_mask = mov_dif>=gamma*mov_std[sign_change_mask] #apply dispersion thresholding
    peaks = np.stack((raw_freq_peaks[stat_peaks_mask], raw_peaks[stat_peaks_mask]),axis=1)  #stack the peaks as [freqs, coordinates]

    if max_peaks:
        if not harm_component:
            raise ValueError('[dispersion_find_peaks] To return the {max_peaks} peaks, a harmonic component must be provided!')
        if len(peaks) >= max_peaks:
            harm_idx = np.argmin(np.abs(raw_freq_peaks[stat_peaks_mask]-harm_component))  # find the index of the harmonic component peak
            return peaks[harm_idx-max_peaks:harm_idx+max_peaks+1] #return the max_peaks peaks on each side of the component
        else:
            raise ValueError(f'[dispersion_find_peaks] length of the peaks matrix is smaller than max_peaks!')
    else:
        return peaks

def fft_dispersion_significant_peaks(data, harm_components, window_size=None, kernel_size=None, gamma=None, mag_threshold=None, max_peaks=None, smooth_kernel=None):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_components: a list/array containing the harmonic components to iterate over
    :param window_size: the size in Hz of the window around each harmonic component (None by default=50Hz)
    :param kernel_size: size of the moving average and std filter kernel (None by default=5)
    :param gamma: threshold for the difference between a peak and the moving average (None by default=1)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-60dB)
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param smooth_kernel: apply curve smoothing to the spectrum if not None, receive a kernel_size
    :return: the coordinates for the six most significant sideband peaks per harmonic component
    len(output) = 3
    output[n] = coordinates
    len(coordinates) <= 6 -> [freq, magnitude]
    '''
    if type(data) != data_types.SimuData:
        raise TypeError(f'[fft_peak_finder] data input must be a SimuData object!')
    if not window_size:
        window_size = 50 #window of 50 Hz around the harmonic spike
    if not kernel_size:
        kernel_size = 5 #set the moving filter kernel to 5x5
    if not gamma:
        gamma = 1 #set the difference threshold as two time the moving standart deviation
    if not mag_threshold:
        mag_threshold = -60 #set the magnitude threshold as -60dB

    #Filter only positive values from the fft frequencies
    freq_mask = data.fft_freqs>=0 #mask to filter negative frequencies
    fft_freqs = data.fft_freqs[freq_mask] #filtered frequencies
    fft_data_dB = data.fft_data_dB[freq_mask] #filtered magnitudes dB

    #Apply curve smoothing if required:
    if smooth_kernel:
        fft_data_dB = np.convolve(fft_data_dB, np.ones(smooth_kernel)/smooth_kernel, mode='same')  #filtered magnitudes dB

    #Store the new values in a PeakFinderData structure to avoid messing with the original data
    fofd = dsp_utils.backwards_num_spectrum_gradient(fft_data_dB, eps=1) #backwards first order finite difference
    finder_data = data_types.PeakFinderData(fft_data_dB, fft_freqs, fofd) #PeakFinderData structure

    #iterate over the harmonic components to find all the significant sideband peaks
    peaks_per_component = [] #list to append the detected peaks for each harmonic component
    for n in harm_components:
        #Window the FFT signal around the peak of the harmonic component
        n = data.fm*n #update the component as a ratio of the fundamental frequency
        lower_idx = np.argmin(np.abs(finder_data.fft_freqs-(n-int(window_size/2)))) #window limit on the left
        upper_idx = np.argmin(np.abs(finder_data.fft_freqs-(n+int(window_size/2)))) #window limit on the right

        #extract the peaks for both window sides
        peaks = dispersion_find_peaks(finder_data, lower_idx, upper_idx, kernel_size=kernel_size,
                                      gamma=gamma, mag_threshold=mag_threshold,
                                      max_peaks=max_peaks, harm_component=n) #peaks at the window
        peaks_per_component.append(peaks) #concatenate the peaks

    return peaks_per_component
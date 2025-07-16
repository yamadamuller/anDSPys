import numpy as np
from framework import data_types, dsp_utils
from scipy.ndimage import gaussian_filter1d
import pandas as pd

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

def dispersion_find_peaks(data, lower_bound_idx, upper_bound_idx, kernel_size=None, gamma=None, mag_threshold=None, h_threshold=None, max_peaks=None, harm_component=None, valley=False):
    '''
    :param data: framework.data_types.PeakFinderData structure
    :param lower_bound_idx: the index of the boundary of the window (lower side)
    :param upper_bound_idx: the index of the boundary of the window (upper side)
    :param kernel_size: size of the moving average and std filter kernel (None by default=5)
    :param gamma: threshold for the difference between a peak and the moving average (None by default=1)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-80dB)
    :param h_threshold: threshold for peak height with respect to its neighboring sample (None by default=0dB)
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param harm_component: the harmonic component to filter max peaks from (only used if max_peaks is not None)
    :param valley: flag to add valleys in the peak detection
    :return: the coordinates for the six most significant sideband peaks of the computed harmonic component
    output[n] = coordinates -> [freqs, magnitudes]
    '''
    if type(data) != data_types.PeakFinderData:
        raise TypeError(f'[dispersion_find_peaks] data input must be a PeakFinderData object!')
    if lower_bound_idx == upper_bound_idx:
        raise ValueError(f'[dispersion_find_peaks] lower and upper boundary of the signal window must not be the same!')
    if not mag_threshold:
        mag_threshold = -60 #set the magnitude threshold as -60dB
    if not h_threshold:
        h_threshold = 0 #set the height threshold as 0dB

    #Find the peaks within the side defined by the boundary index (bound_idx)
    if not data.smoothed:
        wind_freqs = data.fft_freqs[lower_bound_idx:upper_bound_idx]
        wind_spectrum = data.fft_data_dB[lower_bound_idx:upper_bound_idx]
        wind_grad = data.fofd[lower_bound_idx:upper_bound_idx] #gradient of the spectrum (first order finite difference)
    else:
        wind_freqs = data.fft_freqs[lower_bound_idx:upper_bound_idx]
        wind_spectrum = data.smooth_fft_data_dB[lower_bound_idx:upper_bound_idx]
        wind_spectrum_rough = data.fft_data_dB[lower_bound_idx:upper_bound_idx] #unsmoothed data to later extract amplitudes
        wind_grad = data.fofd[lower_bound_idx:upper_bound_idx] #gradient of the spectrum (first order finite difference)

    #Compute the moving filters
    stat_filtering = False #flag to monitor if statistical peak filtering will be apllied
    if (kernel_size is not None) & (gamma is not None):
        mov_avg = np.convolve(wind_spectrum, np.ones(kernel_size)/kernel_size, mode='same') #moving average
        mov_std = apply_moving_filter(wind_spectrum,np.std,kernel_size) #moving standard deviation
        stat_filtering = True #set statistical filtering to True

    #Compute lagged and advanced signals to avoid over-computing
    lag_wind_spectrum = np.roll(wind_spectrum, 1) #lag the spectrum in one sample
    adv_wind_spectrum = np.roll(wind_spectrum, -1) #advance the spectrum in one sample

    #Evaluate signal change in the first derivative to infer local maxima
    grad_sign = np.sign(wind_grad) #compute the signs of each value of the first derivative
    grad_sign_change = np.roll(grad_sign,-1)+grad_sign #signs[i+1]-signs[i]

    #Evaluate spectrum peaks based on its neighbourhood
    if not valley:
        wind_l_upeaks = wind_spectrum>lag_wind_spectrum #check if a sample up peak is greater than its left neighbour
        wind_r_upeaks = wind_spectrum>adv_wind_spectrum #check if a sample up peak is greater its right neighbour
        neighbour_mask = wind_l_upeaks&wind_r_upeaks #check for samples where the values are greater than its neighbour and its an upeak
    else:
        wind_l_upeaks = wind_spectrum>lag_wind_spectrum  #check if a sample up peak is greater than its left neighbour
        wind_l_dpeaks = np.abs(wind_spectrum)>np.abs(lag_wind_spectrum)  #check if a sample down peak is greater than its left neighbour
        wind_r_upeaks = wind_spectrum>adv_wind_spectrum  #check if a sample up peak is greater its right neighbour
        wind_r_dpeaks = np.abs(wind_spectrum)>np.abs(adv_wind_spectrum)  #check if a sample down peaks is greater its right neighbour
        neighbour_mask = (wind_l_upeaks|wind_l_dpeaks)&(wind_r_upeaks|wind_r_dpeaks)  #check for samples where the values are greater than its neighbour

    #Evaluate height change in the spectrum to infer local maxima
    height_diff = np.abs(wind_spectrum-lag_wind_spectrum) #absolute value of the height change
    height_mask = height_diff>=h_threshold #check for samples where its value is greater than its left neighbour

    #Extract the peaks
    if not data.smoothed:
        sign_change_mask = grad_sign_change == 0  #when the first derivative changes from + to -. the sum is 0
        mag_thresh_mask = wind_spectrum >= mag_threshold  #values of the FFT that surpass the magnitude threshold
        sign_change_mask = neighbour_mask & sign_change_mask & mag_thresh_mask & height_mask  #update the mask where all prior masks are valid
        raw_peaks = wind_spectrum[sign_change_mask]  #every peak magnitude detected by the change of signal in the gradient
        raw_freq_peaks = wind_freqs[sign_change_mask]  #every peak frequency detected by the change of signal in the gradient
        if stat_filtering:
            mov_dif = np.abs(raw_peaks - mov_avg[sign_change_mask])  #difference between a sample and the moving mean at its index
            stat_peaks_mask = mov_dif >= gamma * mov_std[sign_change_mask]  #apply dispersion thresholding
            peaks = np.stack((raw_freq_peaks[stat_peaks_mask], raw_peaks[stat_peaks_mask]),axis=1)  #stack the peaks as [freqs, coordinates]
        else:
            peaks = np.stack((raw_freq_peaks, raw_peaks), axis=1) #stack the peaks as [freqs, coordinates]
    else:
        #Extract the peaks
        sign_change_mask = grad_sign_change == 0  #when the first derivative changes from + to -. the sum is 0
        mag_thresh_mask = wind_spectrum >= mag_threshold  #values of the smoothed FFT that surpass the magnitude threshold
        sign_change_mask = neighbour_mask & sign_change_mask & mag_thresh_mask & height_mask  #update the mask where all prior masks are valid
        raw_peaks = wind_spectrum[sign_change_mask]  #every smoothed peak magnitude detected by the change of signal in the gradient
        raw_peaks_rough = wind_spectrum_rough[sign_change_mask] #every unsmoothed peak magnitude detected by the change of signal in the gradient
        raw_freq_peaks = wind_freqs[sign_change_mask]  #every peak frequency detected by the change of signal in the gradient
        if stat_filtering:
            mov_dif = np.abs(raw_peaks_rough - mov_avg[sign_change_mask])  #difference between a sample and the moving mean at its index
            stat_peaks_mask = mov_dif >= gamma * mov_std[sign_change_mask]  #apply dispersion thresholding
            peaks = np.stack((raw_freq_peaks[stat_peaks_mask], raw_peaks[stat_peaks_mask]), axis=1)  #stack the peaks as [freqs, coordinates]
        else:
            peaks = np.stack((raw_freq_peaks, raw_peaks), axis=1)  # stack the peaks as [freqs, coordinates]

    if max_peaks:
        if not harm_component:
            raise ValueError('[dispersion_find_peaks] To return the {max_peaks} peaks, a harmonic component must be provided!')
        if len(peaks) >= max_peaks:
            if stat_filtering:
                harm_idx = np.argmin(np.abs(raw_freq_peaks[stat_peaks_mask]-harm_component))  #find the index of the harmonic component peak
            else:
                harm_idx = np.argmin(np.abs(raw_freq_peaks-harm_component))  # find the index of the harmonic component peak
            return peaks[harm_idx-max_peaks:harm_idx+max_peaks+1] #return the max_peaks peaks on each side of the component
        else:
            raise ValueError(f'[dispersion_find_peaks] length of the peaks matrix is smaller than max_peaks!')
    else:
        return peaks

def distance_find_peaks(data, lower_bound_idx, upper_bound_idx, mag_threshold=None, h_threshold=None, min_peak_dist=None, max_peaks=None, harm_component=None, valley=False):
    '''
    :param data: framework.data_types.PeakFinderData structure
    :param lower_bound_idx: the index of the boundary of the window (lower side)
    :param upper_bound_idx: the index of the boundary of the window (upper side)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-80dB)
    :param h_threshold: threshold for peak height with respect to its neighboring sample (None by default=0dB)
    :param min_peak_dist: filter for the tallest peaks distanced by at least the passed value (None by default=2.s.fm)
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param harm_component: the harmonic component to filter max peaks from (only used if max_peaks is not None)
    :param valley: flag to add valleys in the peak detection
    :return: the coordinates for the six most significant sideband peaks of the computed harmonic component
    output[n] = coordinates -> [freqs, magnitudes]
    '''
    if type(data) != data_types.PeakFinderData:
        raise TypeError(f'[distance_find_peaks] data input must be a PeakFinderData object!')
    if lower_bound_idx == upper_bound_idx:
        raise ValueError(f'[distance_find_peaks] lower and upper boundary of the signal window must not be the same!')
    if not mag_threshold:
        mag_threshold = -60 #set the magnitude threshold as -60dB
    if not h_threshold:
        h_threshold = 0 #set the height threshold as 0dB
    if not min_peak_dist:
        min_peak_dist = 2*data.slip*data.fm #the expected distance between peaks given the slope (2.s.fm)

    #Find the peaks within the side defined by the boundary index (bound_idx)
    wind_freqs = data.fft_freqs[lower_bound_idx:upper_bound_idx]
    wind_spectrum = data.fft_data_dB[lower_bound_idx:upper_bound_idx]
    wind_grad = data.fofd[lower_bound_idx:upper_bound_idx] #gradient of the spectrum (first order finite difference)

    #Compute lagged and advanced signals to avoid over-computing
    lag_wind_spectrum = np.roll(wind_spectrum, 1) #lag the spectrum in one sample
    adv_wind_spectrum = np.roll(wind_spectrum, -1) #advance the spectrum in one sample

    #Evaluate signal change in the first derivative to infer local maxima
    grad_sign = np.sign(wind_grad) #compute the signs of each value of the first derivative
    grad_sign_change = np.roll(grad_sign,-1)+grad_sign #signs[i+1]-signs[i]

    #Evaluate spectrum peaks based on its neighbourhood
    if not valley:
        wind_l_upeaks = wind_spectrum>lag_wind_spectrum #check if a sample up peak is greater than its left neighbour
        wind_r_upeaks = wind_spectrum>adv_wind_spectrum #check if a sample up peak is greater its right neighbour
        neighbour_mask = wind_l_upeaks&wind_r_upeaks #check for samples where the values are greater than its neighbour and its an upeak
    else:
        wind_l_upeaks = wind_spectrum>lag_wind_spectrum  #check if a sample up peak is greater than its left neighbour
        wind_l_dpeaks = np.abs(wind_spectrum)>np.abs(lag_wind_spectrum)  #check if a sample down peak is greater than its left neighbour
        wind_r_upeaks = wind_spectrum>adv_wind_spectrum  #check if a sample up peak is greater its right neighbour
        wind_r_dpeaks = np.abs(wind_spectrum)>np.abs(adv_wind_spectrum)  #check if a sample down peaks is greater its right neighbour
        neighbour_mask = (wind_l_upeaks|wind_l_dpeaks)&(wind_r_upeaks|wind_r_dpeaks)  #check for samples where the values are greater than its neighbour

    #Evaluate height change in the spectrum to infer local maxima
    height_diff = np.abs(wind_spectrum-lag_wind_spectrum) #absolute value of the height change
    height_mask = height_diff>=h_threshold #check for samples where its value is greater than its left neighbour

    #Extract the peaks
    sign_change_mask = grad_sign_change == 0  #when the first derivative changes from + to -. the sum is 0
    mag_thresh_mask = wind_spectrum >= mag_threshold  #values of the FFT that surpass the magnitude threshold
    sign_change_mask = neighbour_mask & sign_change_mask & mag_thresh_mask & height_mask  #update the mask where all prior masks are valid
    raw_peaks = wind_spectrum[sign_change_mask]  #every peak magnitude detected by the change of signal in the gradient
    raw_freq_peaks = wind_freqs[sign_change_mask]  #every peak frequency detected by the change of signal in the gradient
    peaks = np.stack((raw_freq_peaks, raw_peaks), axis=1) #stack the peaks as [freqs, coordinates]

    #Extract the tallest peak every (min_peak_dist) window
    dist_peaks = [] #list to append the distanced peaks
    space_search = np.arange(wind_freqs[0], wind_freqs[-1]+min_peak_dist, min_peak_dist) #divide the frequency window into (min_peak_dist) spaces
    #TODO: find a way to vectorize this operation!
    for i in range(len(space_search)-1):
        l_freq_bound = space_search[i] #lower frequency boundary inside the search space
        u_freq_bound = space_search[i+1] #upper frequency boundary inside the search space
        dist_mask = (peaks[:,0]>=l_freq_bound)&(peaks[:,0]<=u_freq_bound) #filter peaks based on the frequency boundaries
        try: #avoid error in case the mask returns empty
            tallest_peak = np.argmax(peaks[dist_mask,1]) #find the tallest peak in the interval
            dist_peaks.append(peaks[dist_mask][tallest_peak,:]) #extract the tallest peak
        except:
            continue #skip if empty mask
    peaks = np.array(dist_peaks) #convert the list into numpy array and commute with peaks

    if max_peaks:
        if not harm_component:
            raise ValueError('[dispersion_find_peaks] To return the {max_peaks} peaks, a harmonic component must be provided!')
        if len(peaks) >= max_peaks:
            harm_idx = np.argmin(np.abs(peaks[:,0]-harm_component))  #find the index of the harmonic component peak
            return peaks[harm_idx-max_peaks:harm_idx+max_peaks+1] #return the max_peaks peaks on each side of the component
        else:
            raise ValueError(f'[dispersion_find_peaks] length of the peaks matrix is smaller than max_peaks!')
    else:
        return peaks

def fft_significant_peaks(data, harm_components, window_size=None, method='distance', kernel_size=None, gamma=None, mag_threshold=None, h_threshold=None, min_peak_dist=None, max_peaks=None, valley=False, gauss_sigma=None):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_components: a list/array containing the harmonic components to iterate over
    :param window_size: the size in Hz of the window around each harmonic component (None by default=50Hz)
    :param method: which method will be used to detect desired peaks (distance by default)
    :param kernel_size: size of the moving average and std filter kernel (None by default=5)
    :param gamma: threshold for the difference between a peak and the moving average (None by default=1)
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (None by default=-60dB)
    :param h_threshold: threshold for peak height with respect to its neighboring sample (None by default=0dB)
    :param min_peak_dist: filter for the tallest peaks distanced by at least the passed value (None by default)
    :param max_peaks: the maximum number of significant peaks to extract (None by default=all)
    :param valley: flag to add valleys in the peak detection
    :param gauss_sigma: apply gaussian smoothing to the spectrum if not None, receive the gaussian standard deviation
    :return: the coordinates for the most significant sideband peaks per harmonic component
    len(output) = 3
    output[n] = coordinates
    len(coordinates) <= 6 -> [freq, magnitude]
    '''
    if (type(data) != data_types.SimuData) & (type(data) != data_types.LabData):
        raise TypeError(f'[fft_peak_finder] data input must be a SimuData/LabData object!')
    if type(data) == data_types.LabData:
        data.slip = 0 #TODO: placeholder value for now!
    if not window_size:
        window_size = 50 #window of 50 Hz around the harmonic spike
    methods_available = ['distance', 'dispersion','combined'] #available methods for peak finding
    if method not in methods_available:
        raise ValueError(f'[fft_significant_peaks] Method {method} no available! Try {methods_available}')
    if method == 'dispersion':
        if (kernel_size is None) | (gamma is None):
            kernel_size = None #revert to None
            gamma = None #revert to None
            print(f'[fft_significant_peaks] Running dispersion-based peak finding without dispersion factor!')
    if not mag_threshold:
        mag_threshold = -60 #set the magnitude threshold as -60dB
    if not h_threshold:
        h_threshold = 0 #set the height threshold as 0dB

    #Filter only positive values from the fft frequencies
    freq_mask = data.fft_freqs>=0 #mask to filter negative frequencies
    fft_freqs = data.fft_freqs[freq_mask] #filtered frequencies
    fft_data_dB = data.fft_data_dB[freq_mask] #filtered magnitudes dB

    #Apply curve smoothing if required:
    if gauss_sigma:
        smooth_fft_data_dB = gaussian_filter1d(fft_data_dB, gauss_sigma) #smoothed magnitudes dB
    else:
        smooth_fft_data_dB = None

    #Store the new values in a PeakFinderData structure to avoid messing with the original data
    fofd = dsp_utils.backwards_num_spectrum_gradient(fft_data_dB, eps=1) #backwards first order finite difference

    #Create the PeakFinder data structure based on the method chosen
    if method == 'dispersion':
        finder_data = data_types.PeakFinderData(fft_data_dB, fft_freqs, fofd, smooth_data=smooth_fft_data_dB)
    else:
        finder_data = data_types.PeakFinderData(fft_data_dB, fft_freqs, fofd, smooth_data=smooth_fft_data_dB,
                                                slip=data.slip, fm=data.fm)

    #iterate over the harmonic components to find all the significant sideband peaks
    peaks_per_component = [] #list to append the detected peaks for each harmonic component
    for n in harm_components:
        #Window the FFT signal around the peak of the harmonic component
        n = data.fm*n #update the component as a ratio of the fundamental frequency
        lower_idx = np.argmin(np.abs(finder_data.fft_freqs-(n-int(window_size/2)))) #window limit on the left
        upper_idx = np.argmin(np.abs(finder_data.fft_freqs-(n+int(window_size/2)))) #window limit on the right

        #extract the peaks for both window sides
        if method == 'dispersion':
            peaks = dispersion_find_peaks(finder_data, lower_idx, upper_idx, kernel_size=kernel_size,
                                          gamma=gamma, mag_threshold=mag_threshold, h_threshold=h_threshold,
                                          max_peaks=max_peaks, harm_component=n,
                                          valley=valley) #peaks at the window
        elif method == 'distance':
            peaks = distance_find_peaks(finder_data, lower_idx, upper_idx,
                                        mag_threshold=mag_threshold, h_threshold=h_threshold,
                                        min_peak_dist=min_peak_dist, max_peaks=max_peaks, harm_component=n,
                                        valley=valley)  # peaks at the window
        peaks_per_component.append(peaks) #concatenate the peaks

    return peaks_per_component

def organize_peak_data(fft_peaks, loads):
    '''
    :param fft_peaks: the list with all the peak coordinates extracted from the detection routine
    :param loads: the list with all the loads used in the peak findings
    :return: the data organized as a pandas dataframe with the frequency displacement value as well
    '''
    #Convert the all peaks list into an array per load test
    peaks_per_load = [] #list to store the arrays
    load_counter = 0
    for load_peaks in fft_peaks:
        curr_peaks = np.empty((1, 4)) #empty array to concatenate iteratively
        for harm_peak in load_peaks:
            freq_disp = np.abs(harm_peak[:, 0] - np.roll(harm_peak[:, 0], -1)) #roll the frequencies in one sample to find displacement
            freq_disp[-1] = 0 #set the last one to 0 to avoid comparing boundary values
            harm_load = np.ones_like(freq_disp)*loads[load_counter] #register which load has been computed
            curr_peaks = np.concat([curr_peaks, np.stack((harm_load, harm_peak[:,0], harm_peak[:,1], freq_disp), axis=0).T]) #concatenate each harmonic component peaks within the load data
        local_frame = pd.DataFrame(curr_peaks[1:], columns=['load', 'freqs', 'fft_mags', 'freq_disp']) #append the peaks to the list
        peaks_per_load.append(local_frame) #save the dataframes in the global list
        load_counter += 1 #increase the load counter

    return pd.concat(peaks_per_load) #concatenate all the processed dataframes into a single one

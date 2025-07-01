import numpy as np

def apply_dB(signal):
    '''
    :param signal: the array with the signal to be converted from amplitude to dB
    :return: 20*log10(abs(signal))
    '''
    return 20*np.log10(np.abs(signal)+1e-8)

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

def sideband_peak_finder(spectrum, freqs, s, fm=60, threshold=0.25):
    '''
    :param spectrum: spectrum of the signal in dB
    :param freqs: frequencies of the spectrum
    :param s: slip of the simulated motor
    :param fm: fundamental frequency [Hz] (60Hz by default)
    :param threshold: limits around the sidebands to find the maximum
    :return: the points of the three interest peaks (left sideband, fundamental and right sideband)
    each point being -> [magnitude, ...] and [frequencies, ...]
    '''
    filt_freq = freqs>=0 #filter for only positive frequencies
    freqs = freqs[filt_freq] #filter frequencies
    spectrum = spectrum[filt_freq] #filter magnitudes
    points = [] #store the peaks

    #Define the sidebands
    l_band = (1-2*s)*fm #left sideband
    l_mask = (freqs>=l_band-threshold)&(freqs<=l_band+threshold) #mask to search for the peak within the left band threshold
    r_band = (1+2*s)*fm #right sideband
    r_mask = (freqs>=r_band-threshold)&(freqs<=r_band+threshold) #mask to search for the peak within the right band threshold

    #Process the left band
    l_band_spec = spectrum[l_mask] #filter the original signal based on the left band mask
    l_band_peak = np.argmax(l_band_spec) #find the index of the maximum value
    l_band_freqs = freqs[l_mask] #filter the frequencies based on the left band mask
    points.append([l_band_spec[l_band_peak].item(), l_band_freqs[l_band_peak].item()])

    #Process the fundamentl frequency
    fund_peak = np.argmax(spectrum) #find the index of the global maximum
    points.append([spectrum[fund_peak].item(), freqs[fund_peak].item()])

    #Process the right band
    r_band_spec = spectrum[r_mask] #filter the original signal based on the right band mask
    r_band_peak = np.argmax(r_band_spec) #find the index of the maximum value
    r_band_freqs = freqs[r_mask] #filter the frequencies based on the right band mask
    points.append([r_band_spec[r_band_peak].item(), r_band_freqs[r_band_peak].item()])

    return points

def sideband_fdm_peak_finder(spectrum, freqs, s, load, fm=60, threshold=0.25):
    '''
    :param spectrum: spectrum of the signal in dB
    :param freqs: frequencies of the spectrum
    :param s: slip of the simulated motor
    :param load: percentage of the load used in the simulation [%]
    :param fm: fundamental frequency [Hz] (60Hz by default)
    :param threshold: limits around the sidebands to find the maximum
    :return: the points of the three interest peaks (left sideband, fundamental and right sideband)
    each point being -> [magnitude, ...] and [frequencies, ...]
    '''
    filt_freq = freqs>=0 #filter for only positive frequencies
    freqs = freqs[filt_freq] #filter frequencies
    spectrum = spectrum[filt_freq] #filter magnitudes
    points = [] #store the peaks

    #Compute the finite difference between the spectrum and itself lagged in 1 sample
    backward_spectrum = np.roll(spectrum, 1) #s[i-1]
    backward_spectrum[0] = spectrum[0] #set the first elem as the same to avoid overshooting
    forward_spectrum = np.roll(spectrum,-1) #x[i+1]
    backward_spectrum[-1] = spectrum[-1] #set the last elem as the same to avoid overshooting

    #TODO: undertand how to compensate for smaller loads
    if load > 50:
        fdm_spectrum = spectrum - backward_spectrum #s[i]-s[i-1]
    else:
        fdm_spectrum = np.abs(spectrum) - np.abs(backward_spectrum) #abs(s[i])-abs(s[i-1])

    #Define the sidebands
    l_band = (1-2*s)*fm #left sideband
    l_mask = (freqs>=l_band-threshold)&(freqs<=l_band+threshold) #mask to search for the peak within the left band threshold
    r_band = (1+2*s)*fm #right sideband
    r_mask = (freqs>=r_band-threshold)&(freqs<=r_band+threshold) #mask to search for the peak within the right band threshold

    #Process the left band
    fdm_l_band_spec = fdm_spectrum[l_mask] #filter the fdm signal based on the left band mask
    l_band_spec = spectrum[l_mask] #filter the original signal based on the left band mask
    l_band_peak = np.argmax(fdm_l_band_spec) #find the index of the maximum value
    l_band_freqs = freqs[l_mask] #filter the frequencies based on the left band mask
    points.append([l_band_spec[l_band_peak].item(), l_band_freqs[l_band_peak].item()])

    #Process the fundamentl frequency
    fund_peak = np.argmax(spectrum) #find the index of the global maximum
    points.append([spectrum[fund_peak].item(), freqs[fund_peak].item()])

    #Process the right band
    fdm_r_band_spec = fdm_spectrum[r_mask] #filter the fdm signal based on the right band mask
    r_band_spec = spectrum[r_mask]  # filter the fdm signal based on the right band mask
    r_band_peak = np.argmax(fdm_r_band_spec) #find the index of the maximum value
    r_band_freqs = freqs[r_mask] #filter the frequencies based on the right band mask
    points.append([r_band_spec[r_band_peak].item(), r_band_freqs[r_band_peak].item()])

    return points


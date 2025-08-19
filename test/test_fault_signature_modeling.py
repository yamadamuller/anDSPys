import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def BRB_IM_model(s, n_periods, fc, fs, K):
  '''
  :param s: slip
  :param n_periods: number of periods
  :param fc: central frequency [Hz]
  :param fs: sampling frequency [Hz]
  :param K: maximum order of the sidebands
  :return the current signal in the presence of one BRB fault and the timestamps
  '''

  #Based on the analytical model for the current signal in the presence of one BRB fault
  #M. Ma, Z. Cao, H. Fu, W. Xu and J. Dai,
  #"Sparse Bayesian Learning Approach for Broken Rotor Bar Fault Diagnosis"
  #IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-10, 2023
  #doi: 10.1109/TIM.2023.3303505.

  #Model constants
  fb = 2*s*fc #twice-slip frequency
  T = n_periods/fc #total elapsed time
  t = np.arange(0, T+(1/fs), 1/fs) #time samples
  n = np.arange(0,len(t),1)
  k = np.arange(-K,K+1,1) #sideband orders
  Ik = 2*np.exp(-3.5*np.abs(k)) #amplitude of the k-th fault order
  phase = np.zeros_like(k) #phase of the k-th fault order
  noise = np.random.normal(0, 1e-1, len(n)) #gaussian noise with avg=0 and std=0.1

  #Discrete IM model under BRB fault
  Ik = Ik[:, np.newaxis] #expand the Ik array in one dimension
  k = k[:,np.newaxis] #expand the k array in one dimension
  n = n[:, np.newaxis].T #expand the transposed n array in one dimension
  phase = phase[:,np.newaxis] #expand the phase array in one dimension
  brb_bands = fc+(k*fb) #influence of the fault signatures in the ideal cossine function
  samples = n/fs #samples based on the sampling frequency
  current_component = 2*np.pi*(brb_bands@samples+phase) #current signal component with brb bands and k-th fault order phase
  signal_euler = np.exp(1j*current_component) #exp(jx) = cosx + jsinx
  y_n = np.sum(Ik*signal_euler, axis=0)+noise  #E_{-K to K} Ik*exp[j(2pi(fc+k.fb)(n/fs)+phase)] + e[n]

  return y_n, t

def compute_FFT(signal, fs, shift=True, apply_dB=True, normalize_by=np.max):
  '''
  :param signal: the current signal to compute the FFT
  :param fs: sampling frequency [Hz]
  :param shift: center the FFT in 0 Hz (True)
  :param apply_dB: return the FFT in amplitude (False) or dB (True)
  :param normalize_by: which function will be used as the normalization factor
  :return the FFT of the current signal and the FFT frequencies
  '''
  #Compute the FFT of the current signal
  fft_freqs = np.linspace(-fs/2, fs/2, len(signal)) #FFT frequencies
  fft = np.fft.fft(signal) #compute the FFT

  if shift:
    fft = np.fft.fftshift(fft) #shift the FFT

  fft = np.abs(fft)/normalize_by(np.abs(fft)) #normalize the FFT

  if apply_dB:
    fft = 20*np.log10(np.abs(fft)+1e-8)

  return fft, fft_freqs

int_periods = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600] #number of periods to evaluate the FFT fault signatures
plt.figure()
leg = []
for period in int_periods:
  y_n, t = BRB_IM_model(0.02, period, 60, 1e3, 1) #current and time
  fft, fft_freqs = compute_FFT(y_n, 1e3) #FFT and frequencies

  #plot
  plt.plot(fft_freqs, fft)
  leg.append(f'n_periods = {period}')

plt.axvline((1-2*0.02)*60, linestyle='dotted', color='black')
plt.axvline((1+2*0.02)*60, linestyle='dotted', color='black')
plt.title('Current spectrum based on the number of periods')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amp. [dB]')
plt.xlim([50,70])
plt.legend(leg)
plt.grid()
plt.show()

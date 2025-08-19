import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from framework import file_csv
import time

#data = loadmat('../data/ia.mat')['ia'].squeeze() #load the current data into numpy array
simu_dir = "../data/1_broken_bar_03082025/i_phase/noR"  #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, 100, 1800, fm=60, n_periods=600, normalize_by=np.max) #read the simulation data
data = simu_data.i_motor

t_init = time.time()
data = data-np.mean(data)
data = data/np.max(data) #normalize by the maximum
fs = simu_data.fs #sampling freq
fc = simu_data.fm #fundamental frequency
res = simu_data.Res #transform resolution
N = len(data) #length of the current data
n = np.arange(1,N+1,1) #frequency order
baseline = np.cos((1/fs)*2*np.pi*n*fc) #current baseline
data = data*baseline #multiply the current by the baseline
data_env = hilbert(data) #envelope of the signal eq. (5)
data = data_env - np.mean(data_env)
data = data/np.max(np.abs(data)) #normalize by the maximum
C_all = np.arange(0,len(data),1) #discrete indexes
f_sample = fc + np.arange(50,70+res,res) #amplitude modulation due to BRB fault eq. (3)

#Algorithm initialization
data = data.reshape(-1,1) #reshape to a 2d array
M = len(data)
N = len(f_sample)
norm_y = np.linalg.norm(data, 'fro')/np.sqrt(M) #froebenius norm of the current
data = data/norm_y #normalize the data by the norm
omega = 1j*2*np.pi*C_all[:,np.newaxis]/fs #complex model exponent eqs. (5,6)
A = np.exp(omega@f_sample[np.newaxis,:])/np.sqrt(M) #eq. (7)
C = 1j*2*np.pi*C_all[:,np.newaxis]/fs@np.ones((1,N)) #eq. (7)
B = C*A #y_hat = Ax
reslu = f_sample[1] - f_sample[0]
mu = 0
delta = np.ones((N,1))
maxitera = 150
alpha = 1
etc = 10
xi = np.ones((N,1))
h = 2e-4
a = 1e-5
b = a
delta_inv = 1/delta
for itera in range(0,maxitera):
    #update mu and sigma
    AHA = A.conj().T@A #conjugate transpose operation eqs. (25,26)
    sigma = np.linalg.solve((alpha*AHA+np.diag(delta.ravel())), np.eye(N)) #eq. (26)
    mu = sigma@(alpha*(A.conj().T@data)) #eq. (25)
    delta = 1/delta_inv #update delta
    sigma_diag = (np.diag(sigma).reshape(-1,1))
    z = (N-1)//2#symmetric sidebands
    part1 = np.abs(mu[z+1:])**2
    part2 = (np.abs(mu[:z+1])**2)[::-1]
    part3 = sigma_diag[z+1:]
    part4 = sigma_diag[:z+1][::-1]
    temp1 = part1+part2+part3+part4
    temp2 = np.abs(mu[z])**2 + sigma_diag[z]

    #eq. (28)
    delta_inv[z+1:] = (-3+2*np.sqrt(9/4+xi[z+1:]**2*temp1))/(xi[z+1:]**2)
    delta_inv[0:z+1] = delta_inv[z+1:][::-1]
    delta_inv[z] = (-1+np.sqrt(1+4*xi[z]**2*temp2))/(xi[z]**2)

    #eq. (29)
    xi[z:] = (-h+np.sqrt(h**2+2*delta_inv[z:]*(h+2)))/(delta_inv[z:])
    xi[0:z+1] = xi[z+1:][::-1]

    #eq. (30)
    residue = data-A@mu
    alpha_old = alpha
    alpha = (a+M)/(b+np.linalg.norm(residue, 'fro')**2+np.sum(AHA.conj()*sigma)).real
    rho_alpha = 0.95
    alpha = (1-rho_alpha)*alpha+rho_alpha*alpha_old

    #eq. (31)
    BHB = B.conj().T@B
    varpi = mu@mu.conj().T+sigma

    #eqs. (35,36)
    P = (BHB.conj() * varpi).real
    v = (mu.conj()*B.conj().T@residue).real
    eq35_sub = (np.diag(B.conj().T@A@sigma)).real
    v = v - eq35_sub.reshape(-1,1)

    #grid gap
    temp_grid = np.sum(v)/np.sum(P)
    increa_grid = np.sign(temp_grid)/1000 * rho_alpha**itera
    f_sample = f_sample+increa_grid
    A = A*np.exp(omega@np.array([[increa_grid]]))
    B = C*A
    Pm = np.sum(mu*mu.conj(),axis=1)
    Pm = Pm[0:(N-1)//2]+Pm[:(N-1)//2:]
    sort_idx = np.argsort(Pm)[::-1] #descendent order
    idx = sort_idx[0:etc]
    idx = np.concatenate((idx,(N-1-idx)))
    BHB = B[:,idx].conj().T@B[:,idx]
    P = (BHB.conj() * varpi[np.ix_(idx,idx)]).real
    v = (mu[idx].conj()).real*(B[:,idx].conj().T@residue)
    eq35_sub = (np.diag(B[:,idx].conj().T @ A @ sigma[:,idx])).real
    v = v - eq35_sub.reshape(-1,1)
    Permatrix = np.concatenate([-np.eye(etc), np.eye(etc)], axis=0)
    P = Permatrix.T@P@Permatrix
    v = Permatrix.T@v
    temp_grid = (v/np.diag(P).reshape(-1,1)).T
    theld = reslu*rho_alpha**itera
    ind_small = np.abs(temp_grid)<theld
    temp_grid[ind_small] = np.sign(temp_grid[ind_small])*theld
    ind_unchang = np.abs(temp_grid)>reslu
    temp_grid[ind_unchang] = np.sign(temp_grid[ind_unchang])*reslu/20
    f_sample[idx] = f_sample[idx] + np.concatenate([-temp_grid[:,0:etc], temp_grid[:,0:etc]], axis=1).ravel()
    A[:,idx] = A[:,idx] * np.exp(omega@ np.concatenate([-temp_grid[:,0:etc], temp_grid[:,0:etc]], axis=1))
    B[:,idx] = C[:,0:len(idx)]*A[:,idx]

    if itera >= maxitera:
        break

#Output
Pm = np.sum(mu*mu.conj(),axis=1)
indsort = np.argsort(f_sample)
f_sample = f_sample[indsort]
Pm = Pm[indsort]
mu = np.abs(mu[indsort])/np.max(np.abs(mu))
log_mu = 20*np.log10(np.abs(mu)+1e-8)
print(f'[SBL_fault_detect] elapsed = {time.time()-t_init}s')

plt.figure(1)
plt.subplot(2,1,1)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB)
plt.xlim([50,70])
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('FFT')
#plt.legend(leg)
plt.grid()
plt.subplot(2,1,2)
log_mu += 50
f_sample -= fc
leg = []
plt.stem(f_sample, log_mu, markerfmt=" ")
plt.xlim([50,70])
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.title('SBL')
#plt.legend(leg)
plt.grid()
plt.show()

#fault_detect
print(f'[SBL_fault_detect] Bands detected aboved 0dB = {len(log_mu[log_mu>0])}')
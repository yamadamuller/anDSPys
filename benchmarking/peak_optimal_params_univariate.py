from framework import parameter_estimation, data_types, file_sensor_mat
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Important runtime variables
wind_size = 24 #size around each component peak
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
harm_comps = [1,5] #harmonic components
tols = 1/10**(np.arange(1,6,1)) #tolerances that will be computed
t_gs = np.zeros((len(tols),1)) #array to store the processing time of gridsearch parameter estimation
t_gss = np.zeros((len(tols),1)) #array to store the processing time of golden section search parameter estimation

#Read the data and compute the FFT and DFT
directory = "../data/benchtesting_PD/experimento_2_carga_100__19200Hz_19200Hz.MAT" #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, ns, fm=fm, n_periods=1550, normalize_by=np.max) #organize the output in a SimuData structure

for tol in range(len(tols)):
    runtime = np.zeros((1,1)) #array to store 10 elapsed times
    for i in range(len(runtime)):
        t_init = time.time()
        opt_height = parameter_estimation.univariate_gs_estimator(data, harm_comps, [0,3], tols[tol], plot=False)
        runtime[i] = time.time()-t_init #store the time

    t_gs[tol] = np.mean(runtime) #store for the tolerance the average of 10 computations

for tol in range(len(tols)):
    runtime = np.zeros((1, 1))  #array to store 10 elapsed times
    for i in range(len(runtime)):
        t_init = time.time()
        opt_height = parameter_estimation.univariate_gss_estimator(data, harm_comps, [0,3], tol=tols[tol])
        runtime[i] = time.time() - t_init  # store the time

    t_gss[tol] = np.mean(runtime)  # store for the tolerance the average of 10 computations

plt.figure(1)
leg = []
plt.semilogy(tols, t_gs, '-o')
leg.append('Gridsearch')
plt.semilogy(tols, t_gss, '-o')
leg.append('Golden section search')
plt.xlabel("Algorithm's resolution")
plt.ylabel("Estimation time (log)")
plt.title("Univariate estimation benchmark")
plt.grid()
plt.legend(leg)
plt.show()

from framework import parameter_estimation, data_types, file_sensor_mat, file_mat, electromag_utils
import numpy as np
import time

#Important runtime variables
wind_size = 24 #size around each component peak
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
harm_comps = [1,5] #harmonic components

#Read the data and compute the FFT and DFT
directory = "../data/benchtesting_PD/experimento_2_carga_100__19200Hz_19200Hz.MAT" #directory with data is located in the directory prior
data = file_sensor_mat.read(directory, 100, ns, fm=fm, n_periods=1550, normalize_by=np.max) #organize the output in a SimuData structure
t_init = time.time()
opt_height = parameter_estimation.univariate_gs_estimator(data, harm_comps, [0,3], 0.01, plot=True)
print(f'[param_estimation] GS yielded opt. height diff. value in {time.time()-t_init} s!')
print(f'[param_estimation] opt. height diff. = {opt_height.opt_value}')

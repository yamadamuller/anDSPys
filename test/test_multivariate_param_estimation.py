from framework import parameter_estimation, data_types
import time

#Important runtime variables
harm_comps = [1,5] #harmonic components

#Read the data and compute the FFT and DFT
directory = "../data/benchtesting_PD/experimento_2_carga_100__19200Hz_19200Hz.MAT" #directory with data is located in the directory prior
t_init = time.time()
opt_values = parameter_estimation.multivariate_gs_estimator(directory, data_types.SensorData, 100, harm_comps,
                                                            [1000,1600], [0,3], 10, 0.1,
                                                            plot=True)
print(f'[param_estimation] GS yielded opt. height diff. value in {time.time()-t_init} s!')
print(f'[param_estimation] opt. height diff. = {opt_values.opt_mult}')
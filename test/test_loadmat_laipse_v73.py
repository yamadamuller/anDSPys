from framework import data_types
import time

#Input arguments
file = '../data/narco/struct_r1b_R1.mat' #directory with data is located in the directory prior
experiment = 5 #number of the experiment
Ts = 1/50e3 #sampling period
torque = 4 #N.m

t_init = time.time()
data = data_types.loadmat_laipse_v7_3(file, torque, experiment, Ts, n_periods=None)
print(f'[loadmat_laipse_v7_3] elapsed time for the whole data: {time.time()-t_init}s')
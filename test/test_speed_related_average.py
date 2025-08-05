from framework import file_sensor_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Load the files
exp_num = 6
n_periods = 1500
qx66_file = f'../data/benchtesting_PD/experimento_{exp_num}_carga_66__19200Hz_19200Hz.MAT'
qx66_data = file_sensor_mat.read(qx66_file,66, 1800, n_periods=n_periods)
qx33_file = f'../data/benchtesting_PD/experimento_{exp_num}_carga_33__19200Hz_19200Hz.MAT'
qx33_data = file_sensor_mat.read(qx33_file,33, 1800, n_periods=n_periods)

#compute the average for all the n_periods
qx66_vel_avg = np.mean(qx66_data.speed_motor)
qx66_tor_avg = np.mean(qx66_data.torque)
print(f'[load 66%] For all periods: avg_vel = {qx66_vel_avg}; avg_torque = {qx66_tor_avg}')
qx33_vel_avg = np.mean(qx33_data.speed_motor)
qx33_tor_avg = np.mean(qx33_data.torque)
print(f'[load 33%] For all periods: avg_vel = {qx33_vel_avg}; avg_torque = {qx33_tor_avg}')

#compute the average for the last N periods
last_periods = 10
qx66_lastNperiods = qx66_data.time_grid[-1] - (last_periods/qx66_data.fm)
qx66_mask = qx66_data.time_grid>=qx66_lastNperiods
qx66_filt_vel_avg = np.mean(qx66_data.speed_motor[qx66_mask])
qx66_filt_tor_avg = np.mean(qx66_data.torque[qx66_mask])
print(f'[load 66%] For the last {last_periods} periods: avg_vel = {qx66_filt_vel_avg}; avg_torque = {qx66_filt_tor_avg}')
qx33_lastNperiods = qx33_data.time_grid[-1] - (last_periods/qx33_data.fm)
qx33_mask = qx33_data.time_grid>=qx33_lastNperiods
qx33_filt_vel_avg = np.mean(qx33_data.speed_motor[qx33_mask])
qx33_filt_tor_avg = np.mean(qx33_data.torque[qx33_mask])
print(f'[load 33%] For the last {last_periods} periods: avg_vel = {qx33_filt_vel_avg}; avg_torque = {qx33_filt_tor_avg}')
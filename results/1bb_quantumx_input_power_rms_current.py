from framework import file_sensor_mat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Load the files
exp_num = 1
n_periods = 1500
qx66_file = f'../data/benchtesting_PD/experimento_{exp_num}_carga_66_19200Hz_19200Hz.MAT'
qx66_data = file_sensor_mat.read(qx66_file,66, 1800, n_periods=n_periods)
qx33_file = f'../data/benchtesting_PD/experimento_{exp_num}_carga_33_19200Hz_19200Hz.MAT'
qx33_data = file_sensor_mat.read(qx33_file,33, 1800, n_periods=n_periods)
qx100_file = f'../data/benchtesting_PD/experimento_{exp_num}_carga_100__19200Hz_19200Hz.MAT'
qx100_data = file_sensor_mat.read(qx100_file,100, 1800, n_periods=n_periods)
fp_66 = 0.71
fp_33 = 0.64
fp_100 = 0.81
a= fp_66*1
#compute the average for all the n_periods
qx66_current_max_r_rms = np.max(qx66_data.i_r)/(2**0.5)
qx66_voltage_max_r_rms = ((np.max(qx66_data.v_r)*(3**0.5))/(2**0.5))
qx66_current_max_s_rms = np.max(qx66_data.i_s)/(2**0.5)
qx66_voltage_max_s_rms = ((np.max(qx66_data.v_s)*(3**0.5))/(2**0.5))
qx66_current_max_t_rms = np.max(qx66_data.i_t)/(2**0.5)
qx66_voltage_max_t_rms = ((np.max(qx66_data.v_t)*(3**0.5))/(2**0.5))
i_media = (qx66_current_max_r_rms+qx66_current_max_s_rms+qx66_current_max_t_rms)/3
v_media = (qx66_voltage_max_r_rms+qx66_voltage_max_s_rms+qx66_voltage_max_t_rms)/3
qx66_input_power= (i_media*v_media)*fp_66*(3**0.5)
qx66_rms_current = (qx66_current_max_r_rms+qx66_current_max_s_rms+qx66_current_max_t_rms)/3
qx66_power_bazzo = (v_media/(3**0.5))*3*i_media*fp_66
print(f'[load 66%] For all periods: input_power = {qx66_input_power}; line_current = {qx66_rms_current}; power_bazzo = {qx66_power_bazzo}')
#compute the average for all the n_periods
qx33_current_max_r_rms = np.max(qx33_data.i_r)/(2**0.5)
qx33_voltage_max_r_rms = ((np.max(qx33_data.v_r)*(3**0.5))/(2**0.5))
qx33_current_max_s_rms = (np.max(qx33_data.i_s))/(2**0.5)
qx33_voltage_max_s_rms = (np.max(qx33_data.v_s)*(3**0.5))/(2**0.5)
qx33_current_max_t_rms = np.max(qx33_data.i_t)/(2**0.5)
qx33_voltage_max_t_rms = ((np.max(qx33_data.v_t)*(3**0.5))/(2**0.5))
i_media_33 = (qx33_current_max_r_rms+qx33_current_max_s_rms+qx33_current_max_t_rms)/3
v_media_33 = (qx33_voltage_max_r_rms+qx33_voltage_max_s_rms+qx33_voltage_max_t_rms)/3
qx33_input_power= i_media_33*v_media_33*(3**0.5)*fp_33
qx33_rms_current = (qx33_current_max_r_rms+qx33_current_max_s_rms+qx33_current_max_t_rms)/3
qx33_power_bazzo = (v_media_33/(3**0.5))*3*i_media_33*fp_33
print(f'[load 33%] For all periods: input_power = {qx33_input_power}; line_current = {qx33_rms_current}; power_bazzo = {qx33_power_bazzo}')
#compute the average for all the n_periods
qx100_current_max_r_rms = np.max(qx100_data.i_r)/(2**0.5)
qx100_voltage_max_r_rms = ((np.max(qx100_data.v_r)*(3**0.5))/(2**0.5))
qx100_current_max_s_rms = np.max(qx100_data.i_s)/(2**0.5)
qx100_voltage_max_s_rms = ((np.max(qx100_data.v_s)*(3**0.5))/(2**0.5))
qx100_current_max_t_rms = np.max(qx100_data.i_t)/(2**0.5)
qx100_voltage_max_t_rms = ((np.max(qx100_data.v_t)*(3**0.5))/(2**0.5))
i_media_100 = (qx100_current_max_r_rms+qx100_current_max_s_rms+qx100_current_max_t_rms)/3
v_media_100 = (qx100_voltage_max_r_rms+qx100_voltage_max_s_rms+qx100_voltage_max_t_rms)/3
qx100_input_power= i_media_100*v_media_100*(3**0.5)*fp_100
qx100_rms_current = (qx100_current_max_r_rms+qx100_current_max_s_rms+qx100_current_max_t_rms)/3
qx100_power_bazzo = (v_media_100/(3**0.5))*3*i_media_100*fp_100
print(f'[load 100%] For all periods: input_power = {qx100_input_power}; line_current = {qx100_rms_current}; power_bazzo = {qx100_power_bazzo}')
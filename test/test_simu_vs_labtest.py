from framework import file_csv, file_sensor_mat, data_types
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Load environmental variables
config_file = data_types.load_config_file('../config_file.yml') #load the config file
ns = int(config_file["motor-configs"]["ns"]) #synchronous speed [rpm]
fm = int(config_file["motor-configs"]["fm"]) #fundamental frequency
loadp = 100 #load percentage

#Reading data for the broken bar simulation
simu_dir = "../data/1_broken_bar_28072025"  #path to the directory with the simulation data
simu_data = file_csv.read(simu_dir, loadp, ns, fm=fm, normalize_by=len) # read the simulation data

#Reading data for the broken bar benchtesting
bench_dir = "../data/benchtesting_PD"  #path to the directory with the bench test data
bench_test = file_sensor_mat.read(bench_dir, loadp, ns, batch=True, normalize_by=len) #read the bench testing file as a batch (all files)

#Plot all the FFTs
plt.figure(1)
leg = []
plt.plot(simu_data.fft_freqs, simu_data.fft_data_dB, color='black')
leg.append('Simulation')
for batch_idx in range(len(bench_test.batch_data)):
    batch_elem = bench_test.batch_data[batch_idx] #extract the SensorData structure at the current index
    plt.plot(batch_elem.fft_freqs, batch_elem.fft_data_dB)
    leg.append(f'Experiment {batch_idx}')
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT magnitude [dB]')
plt.xlim([50,70])
plt.grid()
plt.legend(leg)
plt.show()
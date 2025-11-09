from framework import dsp_utils, data_types, file_sensor_mat, file_csv, file_mat, electromag_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class OptResults:
    '''
    This class generates a data structure to store all the outputs from the optimization procedures
    depending on the available computed values
    '''

    def __init__(self, opt_value=None, opt_mult=None, opt_fun=None, jac=None, hess=None, n_iter=None):
        if opt_value is not None:
            self.opt_value = opt_value
        if opt_mult is not None:
            self.opt_mult = opt_mult
        if opt_fun is not None:
            self.opt_fun = opt_fun
        if jac is not None:
            self.jac = jac
        if hess is not None:
            self.hess = hess
        if n_iter is not None:
            self.n_iter = n_iter

def custom_SSE(data, harm_comps, candidate_value, mag_threshold=-80, k_real=14):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_comps: list with the harmonic components that will be searched for peaks
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (default=-80dB)
    :param k_real: expected number of peaks to be detected (default=14)
    :return: sum of all squared errors + the squared error between number of peaks and expected number
             -> \Psi(\theta) = \sum_{i=0}^{\hat{k}-1}(f[i] - \hat{f}_{\theta}[i])^{2} + (k_{real} - \hat{k})^{2}
    '''

    peaks = dsp_utils.fft_significant_peaks(data, harm_comps, method='distance', mag_threshold=mag_threshold,
                                            h_threshold=candidate_value) #run the peak detection routine
    sqr_error = 0 #variable to monitor the squared error between frequencies
    k_hat = 0 #variable to monitor the number of detected peaks per harmonic
    v = np.array([1,5])[np.newaxis, :].T #harmonic components
    k = np.arange(-3,4,1)[np.newaxis, :] #fault orders
    analytical_signature = (v+2*k*data.slip)*data.fm #expected fault signatures location
    harm_counter = 0
    for harm_peak in peaks:
        argmin_freq_diff = harm_peak[:,0][np.newaxis,:].T - analytical_signature[harm_counter,:][np.newaxis,:] #subtract every detected peak by every expected peak (analytical)
        argmin_freq_diff = np.argmin(np.abs(argmin_freq_diff), axis=1) #find the index per line of the minimum difference
        argmin_relative = analytical_signature[harm_counter, argmin_freq_diff] #the expected frequencies of the detected peaks
        sqr_error += np.sum((harm_peak[:, 0] - argmin_relative)**2) #squared error between detected and expected frequencies
        k_hat += len(harm_peak) #updated the detected peaks counter
        harm_counter += 1 #increase the harmonic counter

    return sqr_error + (k_real-k_hat)**2 #sum of all squared errors + the squared error between number of peaks and expected number

def univariate_gs_estimator(data, harm_comps, bounds, step, mag_threshold=-80, k_real=14, plot=True):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_comps: list with the harmonic components that will be searched for peaks
    :param bounds: a list containing the boundaries used to compute the grid search
    :param step: step between candidate values
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (default=-80dB)
    :param k_real: expected number of peaks to be detected (default=14)
    :param plot: flag to administer the plot of the cost values of the grid search (True by default)
    :return:
        optimal_value: the candidate that yielded the smallest cost function value in the grid search
        optimal_cost: the smallest cost computed with the optimal value
        n_iterations: the number of iterations of the algorithm
    '''
    if ((type(data) != data_types.SimuData) &
            (type(data) != data_types.SensorData) &
            (type(data) != data_types.NIHardwareData) &
            (type(data) != data_types.LaipseData)):
        raise TypeError(f'[univariate_gs_estimator] data input must be a SimuData/SensorData/NIHardwareData/LaipseData object!')
    if ((type(harm_comps) != list) &
            (type(harm_comps)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] harm comps must be a list or array, current type is {type(harm_comps)}!')
    if ((type(bounds) != list) &
            (type(bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] bounds must be a list or array, current type is {type(bounds)}!')
    if bounds[-1]<=bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper limit must be greater than the lower limit!')
    if k_real<0:
        raise ValueError('[univariate_gs_estimator] k_real must be >= 0!')

    candidate_values = np.arange(bounds[0], bounds[-1]+step, step) #array of the candidate values
    cost_values = np.zeros((len(candidate_values), 1)) #array to store the computed costs per candidates

    for thetas in range(len(candidate_values)):
        cost_values[thetas] = custom_SSE(data, harm_comps, candidate_values[thetas], mag_threshold=mag_threshold, k_real=k_real)

    theta = np.argmin(cost_values) #find the index of the optimal value

    if plot:
        leg = []
        plt.figure(1)
        plt.plot(candidate_values, cost_values)
        leg.append('Grid search')
        plt.scatter(candidate_values[theta], cost_values[theta], marker='x', color='tab:orange')
        leg.append('Opt. value')
        plt.xlabel("Candidate height diff. values")
        plt.ylabel("Cost function")
        plt.legend(leg)
        plt.grid()
        plt.show()

    return OptResults(opt_value=candidate_values[theta], opt_fun=cost_values[theta], n_iter=len(candidate_values))

def univariate_gss_estimator(data, harm_comps, bounds, tol=1e-6, mag_threshold=-80, k_real=14):
    '''
    :param data: framework.data_types.SimuData structure
    :param harm_comps: list with the harmonic components that will be searched for peaks
    :param bounds: a list containing the boundaries used to compute the grid search
    :param tol: tolerance of the algorithm
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (default=-80dB)
    :param k_real: expected number of peaks to be detected (default=14)
    :return:
        optimal_value: the candidate that yielded the smallest cost function value in the grid search
        optimal_cost: the smallest cost computed with the optimal value
        n_iterations: the number of iterations of the algorithm
    '''
    if ((type(data) != data_types.SimuData) &
            (type(data) != data_types.SensorData) &
            (type(data) != data_types.NIHardwareData) &
            (type(data) != data_types.LaipseData)):
        raise TypeError(f'[univariate_gs_estimator] data input must be a SimuData/SensorData/NIHardwareData/LaipseData object!')
    if ((type(harm_comps) != list) &
            (type(harm_comps)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] harm comps must be a list or array, current type is {type(harm_comps)}!')
    if ((type(bounds) != list) &
            (type(bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] bounds must be a list or array, current type is {type(bounds)}!')
    if bounds[-1]<=bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper limit must be greater than the lower limit!')
    if k_real<0:
        raise ValueError('[univariate_gs_estimator] k_real must be >= 0!')

    #Golden Section Search
    c = (-1 + np.sqrt(5))/2 #constant reduction rate
    c_sub = 1-c #constant reduction rate subtracted from 1
    x1 = bounds[0]*c + bounds[1]*c_sub #first guess of x1
    x2 = bounds[1]*c + bounds[0]*c_sub #first guess of x2
    h = bounds[1] - bounds[0] #forward error
    n = int(np.ceil(np.log(tol/h)/np.log(c))) #iterations to convergence
    for i in range(n):
        fx1 = custom_SSE(data, harm_comps, x1, mag_threshold=mag_threshold, k_real=k_real) #evaluate the cost func at x1
        fx2 = custom_SSE(data, harm_comps, x2, mag_threshold=mag_threshold, k_real=k_real) #evaluate the cost func at x2

        if fx1 < fx2:
            bounds[1] = x2 #update the upper bound given the function at the lower bound is smaller
            x2 = x1 #commute the arguments
            x1 = bounds[0]*c + bounds[1]*c_sub #update x1
        else:
            bounds[0] = x1 #update the lower bound given the function at the upper bound is smaller
            x1 = x2 #commute the arguments
            x2 = bounds[1]*c + bounds[0]*c_sub #update x2

    theta = (bounds[1]+bounds[0])/2 #compute the optimum value as the average between the bounds

    return OptResults(opt_value=theta,
                      opt_fun=custom_SSE(data, harm_comps, theta, mag_threshold=mag_threshold, k_real=k_real),
                      n_iter=n)

def multivariate_gs_estimator(file, data_type, load, harm_comps, p1_bounds, p2_bounds, p1_step, p2_step, mag_threshold=-80, k_real=14, normalize_by=np.max, plot=True):
    '''
    :param file: path to the output file in the local filesystem
    :param harm_comps: list with the harmonic components that will be searched for peaks
    :param p1_bounds: a list containing the boundaries used to compute the grid search for the first parameter
    :param p2_bounds: a list containing the boundaries used to compute the grid search for the second parameter
    :param p1_step: step between p1 candidate values
    :param p2_step: step between p2 candidate values
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (default=-80dB)
    :param k_real: expected number of peaks to be detected (default=14)
    :param normalize_by: which function will be used to normalize the FFT
    :param plot: flag to administer the plot of the cost values of the grid search (True by default)
    :return:
        optimal_value: the candidate values that yielded the smallest cost function value in the grid search
        optimal_cost: the smallest cost computed with the optimal values
        n_iterations: the number of iterations of the algorithm
    '''
    if ((type(harm_comps) != list) &
            (type(harm_comps)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] harm comps must be a list or array, current type is {type(harm_comps)}!')
    if ((type(p1_bounds) != list) &
            (type(p1_bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] pq_bounds must be a list or array, current type is {type(p1_bounds)}!')
    if ((type(p2_bounds) != list) &
            (type(p2_bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] pq_bounds must be a list or array, current type is {type(p2_bounds)}!')
    if p1_bounds[-1]<=p1_bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper p1 limit must be greater than the lower limit!')
    if p2_bounds[-1]<=p2_bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper p2 limit must be greater than the lower limit!')
    if k_real<0:
        raise ValueError('[univariate_gs_estimator] k_real must be >= 0!')

    config_file = data_types.load_config_file('../config_file.yml') #load the config file
    p1_candidates = np.arange(p1_bounds[0], p1_bounds[-1]+p1_step, p1_step) #n_periods candidate values
    p2_candidates = np.arange(p2_bounds[0], p2_bounds[-1]+p2_step, p2_step) #height diff. candidate values
    cost_values = np.zeros((len(p2_candidates), len(p1_candidates))) #x-axis = n_periods / y-axis = height diffs.
    for p1_theta in range(len(p1_candidates)):
        if data_type == data_types.SimuData:
            data = file_csv.read(file, load, int(config_file["motor-configs"]["ns"]), int(config_file["motor-configs"]["fm"]), n_periods=p1_candidates[p1_theta], normalize_by=normalize_by) #organize the output in a SimuData structure
        elif data_type == data_types.SensorData:
            data = file_sensor_mat.read(file, load, int(config_file["motor-configs"]["ns"]), int(config_file["motor-configs"]["fm"]), n_periods=p1_candidates[p1_theta], normalize_by=normalize_by) #organize the output in a SensorData structure
        elif data_type == data_types.LaipseData:
            #TODO: remove hardcoded parameters
            data = file_mat.read(file, 4., fs = 50e3, n_periods=p1_candidates[p1_theta], exp_num=5, normalize_by=normalize_by) #organize the output in a LaipseData structure
            data.slip = electromag_utils.compute_slip(1800, 1732.5)
        else:
            raise TypeError(f'[multivariate_gs_estimator] {data_type} is not valid!')

        for p2_theta in range(len(p2_candidates)):
            cost_values[p2_theta, p1_theta] = custom_SSE(data, harm_comps, p2_candidates[p2_theta], mag_threshold=mag_threshold, k_real=k_real)

    theta = np.where(cost_values == np.min(cost_values)) #find the indexes of the optimum value

    if plot:
        plt.figure(1)
        leg = []
        plt.imshow(cost_values, aspect='auto', extent=[p1_candidates[0], p1_candidates[-1], p2_candidates[-1], p2_candidates[0]])
        plt.scatter(p1_candidates[theta[1][0]], p2_candidates[theta[0][0]], marker='x', color='tab:orange')
        leg.append('Multivariate Grid Search')
        plt.xlabel("Candidate n_period values")
        plt.ylabel("Candidate height diff. values")
        plt.legend(leg)
        plt.show()

    return OptResults(opt_mult=[p2_candidates[theta[0][0]], p1_candidates[theta[1][0]]],
                       opt_fun=cost_values[theta[0][0], theta[1][0]],
                       n_iter=len(p1_candidates)*len(p2_candidates))

def multivariate_pseudo_gs_estimator(file, data_type, load, harm_comps, p1_bounds, p2_bounds, p1_step, tol=1e-6, mag_threshold=-80, k_real=14, normalize_by=np.max):
    '''
    :param file: path to the output file in the local filesystem
    :param harm_comps: list with the harmonic components that will be searched for peaks
    :param p1_bounds: a list containing the boundaries used to compute the grid search for the first parameter
    :param p2_bounds: a list containing the boundaries used to compute the grid search for the second parameter
    :param p1_step: step between p1 candidate values
    :param tol: tolerance of the gss algorithm
    :param mag_threshold: threshold for magnitude values of the FFT [dB] (default=-80dB)
    :param k_real: expected number of peaks to be detected (default=14)
    :param normalize_by: which function will be used to normalize the FFT
    :return:
        optimal_value: the candidate values that yielded the smallest cost function value in the grid search
        optimal_cost: the smallest cost computed with the optimal values
        n_iterations: the number of iterations of the algorithm
    '''
    if ((type(harm_comps) != list) &
            (type(harm_comps)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] harm comps must be a list or array, current type is {type(harm_comps)}!')
    if ((type(p1_bounds) != list) &
            (type(p1_bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] pq_bounds must be a list or array, current type is {type(p1_bounds)}!')
    if ((type(p2_bounds) != list) &
            (type(p2_bounds)!= np.array)):
        raise TypeError(f'[univariate_gs_estimator] pq_bounds must be a list or array, current type is {type(p2_bounds)}!')
    if p1_bounds[-1]<=p1_bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper p1 limit must be greater than the lower limit!')
    if p2_bounds[-1]<=p2_bounds[0]:
        raise ValueError('[univariate_gs_estimator] Upper p2 limit must be greater than the lower limit!')
    if k_real<0:
        raise ValueError('[univariate_gs_estimator] k_real must be >= 0!')

    config_file = data_types.load_config_file('../config_file.yml') #load the config file
    p1_candidates = np.arange(p1_bounds[0], p1_bounds[-1]+p1_step, p1_step) #n_periods candidate values
    cost_values = np.zeros((len(p1_candidates), 1)) #array to store the optimal combination for each p1 candidate based on the gss
    gss_values = np.zeros_like(cost_values) #array to store the optimum value of the heights for each gss
    for p1_theta in range(len(p1_candidates)):
        if data_type == data_types.SimuData:
            data = file_csv.read(file, load, int(config_file["motor-configs"]["ns"]), int(config_file["motor-configs"]["fm"]), n_periods=p1_candidates[p1_theta], normalize_by=normalize_by) #organize the output in a SimuData structure
        elif data_type == data_types.SensorData:
            data = file_sensor_mat.read(file, load, int(config_file["motor-configs"]["ns"]), int(config_file["motor-configs"]["fm"]), n_periods=p1_candidates[p1_theta], normalize_by=normalize_by) #organize the output in a SensorData structure
        elif data_type == data_types.LaipseData:
            #TODO: remove hardcoded parameters
            data = file_mat.read(file, 4., fs = 50e3, n_periods=p1_candidates[p1_theta], exp_num=5, normalize_by=normalize_by) #organize the output in a LaipseData structure
            data.slip = electromag_utils.compute_slip(1800, 1732.5)
        else:
            raise TypeError(f'[multivariate_gs_estimator] {data_type} is not valid!')

        optimal_p2 = univariate_gss_estimator(data, harm_comps, p2_bounds, tol=tol, mag_threshold=mag_threshold, k_real=k_real) #compute the gss for each p1 candidate
        gss_values[p1_theta] = optimal_p2.opt_value #store the argument
        cost_values[p1_theta] = optimal_p2.opt_fun #store the cost of the argument

    theta = np.argmin(cost_values) #find the indexes of the optimum value

    return OptResults(opt_mult=[gss_values[theta], p1_candidates[theta]],
                      opt_fun=cost_values[theta],
                      n_iter=len(p1_candidates)*optimal_p2.n_iter)
import numpy as np

"""
Input:
    spike_times (ndarray): spike times produced by output neurons
                                    shape: [n_samples, num_classes]
    target_classes (ndarray): the correct class of each input (shape: [n_samples])
    num_classes (int): total number of classes
    t_max (int): max time of simulation (paper: 255)
    tau (int): desired time for the correct neuron (paper: 20)

The function compute for each sample the error between the firing times of it's ouutpun neurons and the desired firing
time of each output neuron, that is t_max if it is the wrong neuron and 20 if it is the correct one.

Returns:
    The total error of the dataset
"""

def temporal_fitness_function(spike_times, target_classes, num_classes, t_max=255, tau=20):

    n_samples = spike_times.shape[0]
    total_error = 0.0

    for i in range(n_samples):

        #create a numpy array of lenght num_classes where each element is equal to t_max
        target_firing_times = np.full((num_classes,), t_max)
        target_firing_times[target_classes[i]] = tau
        
        #calculate the error of this sample and add to the total error
        errors = (spike_times[i] - target_firing_times) / t_max
        squared_error = 0.5 * np.sum(errors ** 2)
        total_error += squared_error

    return total_error / n_samples
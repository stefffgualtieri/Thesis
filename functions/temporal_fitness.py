import numpy as np
import torch
'''
Input:
- spike_times (torch.Tensor): shape (n_samples, num_classes), spike times of output neurons.
- target_classes (torch.Tensor): shape (n_samples,), true class indices.
- num_classes (int): number of output neurons/classes.
- t_max (int): maximum simulation time.
- tau (int): desired firing time for target neuron.
- device (str): 'cpu' or 'cuda'.

Returns:
- torch.Tensor: scalar, mean temporal error across samples
'''


def temporal_fitness_function(spike_times, target_classes, num_classes, t_max=255, tau=20, device="cpu"):

    n_samples = spike_times.shape[0]

    #Copstruisci target firing times: default t_max
    target_firing_times = torch.full((n_samples, num_classes), t_max, device=device)

    #Tau for the neurons of the correct class
    target_firing_times[torch.arange(n_samples), target_classes] = tau

    #calcola gli errori normalizzati
    errors = (spike_times - target_firing_times)/t_max

    #Errore quadrato per ogni campione
    squared_error = 0.5 * torch.sum(errors ** 2, dim = 1)

    return squared_error.mean()


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

# def temporal_fitness_function(spike_times, target_classes, num_classes, t_max=255, tau=20):

#     n_samples = spike_times.shape[0]
#     total_error = 0.0

#     for i in range(n_samples):

#         #create a numpy array of lenght num_classes where each element is equal to t_max
#         target_firing_times = np.full((num_classes,), t_max)
#         target_firing_times[target_classes[i]] = tau
        
#         #calculate the error of this sample and add to the total error
#         errors = (spike_times[i] - target_firing_times) / t_max
#         squared_error = 0.5 * np.sum(errors ** 2)
#         total_error += squared_error

#     return total_error / n_samples
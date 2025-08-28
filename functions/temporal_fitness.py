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


def temporal_fitness_function(spike_times, target_classes, t_sim=255, tau=20, device="cpu"):

    n_samples, num_classes = spike_times.shape

    #Copstruisci  un tensore con dentro i target firing times (di default inseriamo t_max)
    target_firing_times = torch.full((n_samples, num_classes), t_sim, device=device)

    #Per ogni riga (=input), seleziona la colonna giusta (=target_class) e sostituisci quel valore con tau
    target_firing_times[torch.arange(n_samples), target_classes] = tau

    #Calcola gli errori normalizzati
    errors = (spike_times - target_firing_times)/t_sim

    #Errore quadrato per ogni campione
    squared_error = 0.5 * torch.sum(errors ** 2, dim = 1)

    return squared_error.mean()
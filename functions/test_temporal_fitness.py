from temporal_fitness import temporal_fitness_function
import torch
import pytest

#Eseguiamo dei test per la funzione di  fitness temporale

def test_temporal_function_example1():
    spike_times = torch.tensor([[12, 255, 245],
                                [214, 30, 240],
                                [200, 210, 17]])
    target_classes = torch.tensor([0, 1, 2])

    error = temporal_fitness_function(spike_times, target_classes)
    expected = 0.01853
    print(error)
    assert torch.isclose(error, torch.tensor(expected), atol=1e-5)


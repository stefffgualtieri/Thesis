import numpy as np
from functions.temporal_fitness import temporal_fitness_function

def test_temporal_fitness_function():

    #2 samples, 3 classes

    spike_times = np.array([[10, 200, 255],
                            [240, 25, 255]], dtype=float)
    target_classes = np.array([0, 1])
    num_classes = 3
    t_max = 255
    tau = 20

    # do it with calculator
    error_expected = 0.02596 / 2

    result = temporal_fitness_function(spike_times, target_classes, num_classes, t_max, tau)

    assert np.isclose(result, error_expected, atol=1e-5)
import numpy as np

def temporal_fitness_function(actual_spike_times, target_class, num_classes, t_max=255, tau=20):
    """
    Calcola l'errore temporale tra i tempi di spike osservati e i target relativi.

    Args:
        actual_spike_times (ndarray): tempi di spike prodotti dai neuroni di output (shape: [num_classes])
        target_class (int): classe corretta per l'input corrente
        num_classes (int): numero totale di classi
        t_max (int): tempo massimo di simulazione
        tau (int): tempo desiderato per il neurone corretto (early spike)

    Returns:
        float: errore quadratico normalizzato Es (da minimizzare)
    """

    target_firing_times = np.full((num_classes,), t_max)
    target_firing_times[target_class] = tau

    errors = (actual_spike_times - target_firing_times)/t_max

    squared_error = 0.5 * np.sum(errors ** 2)
    return squared_error

if __name__ == "__main__":
    # Esempio di test
    actual = np.array([32.1, 240.0, 250.0])
    target = 0  # classe corretta
    print("Errore temporale:", temporal_fitness_function(actual, target, num_classes=3))

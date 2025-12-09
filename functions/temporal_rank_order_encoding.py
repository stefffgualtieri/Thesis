import numpy as np

def temporal_rank_order_encode(X, x_min, x_max, T, eps =1e-8):
    """
    X:  [N, F] numpy
    x_min, x_max: [F] (calcolati sul train)
    T:  numero di time-step (t va in [0, T-1])

    Restituisce: spike_times [N, F] interi in [0, T-1]
                 valori grandi -> t piccolo (spike precoce)
    """
    # normalizza feature per feature in [0,1]
    x_norm = (X - x_min) / (x_max - x_min + eps)   # [N, F]

    # mapping decrescente: 1 -> t_min, 0 -> t_max
    t = (T - 1) * (1.0 - x_norm)                  # [N, F] float
    t = np.rint(t).astype(np.int64)               # arrotonda a intero
    t = np.clip(t, 0, T - 1)

    return t
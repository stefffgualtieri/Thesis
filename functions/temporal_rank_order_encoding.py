import numpy as np

def temporal_rank_order_encode(X, x_min, x_max, T, eps =1e-8):
    """
    - X:  [N, F]
    - x_min, x_max: [F]
    - T:  number of time-steps (t va in [0, T-1])

    - Returns: spike_times [N, F], integer values [0, T-1]
                 big value -> small t
    """
    # Normalization in [0, 1]
    x_norm = (X - x_min) / (x_max - x_min + eps)   # [N, F]

    # mapping: 1 -> t_min, 0 -> t_max
    t = (T - 1) * (1.0 - x_norm)                  # [N, F] float
    t = np.rint(t).astype(np.int64)               # arrotonda a intero
    t = np.clip(t, 0, T - 1)

    return t
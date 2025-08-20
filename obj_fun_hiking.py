import numpy as np
from temporal_fitness import temporal_fitness_function

def make_obj_fun(X_train, y_train, num_classes, t_max=255, tau=20, lambda_l2=1e-6, use_bias=True):
    """
    Ritorna:
      - obj_fun(w) -> float   (la funzione che passerai a hiking_optimization)
      - dim        -> dimensione del vettore w atteso
      - predict_times(w, X)   -> utility per valutare sul test
    """
    n_features = X_train.shape[1]
    dim = n_features * num_classes + (num_classes if use_bias else 0)

    def unpack(w):
        if use_bias:
            W = w[: n_features * num_classes].reshape(n_features, num_classes)
            b = w[n_features * num_classes :].reshape(1, num_classes)
        else:
            W = w.reshape(n_features, num_classes)
            b = np.zeros((1, num_classes))
        return W, b

    def predict_times(w, X):
        W, b = unpack(w)
        W = np.clip(W, 0.0, None)                 # stabilità
        T = X @ W + b
        return np.clip(T, 0.0, float(t_max))      # tempi nell’intervallo

    def obj_fun(w):
        # difesa su dimensione sbagliata
        if w.ndim != 1 or w.shape[0] != dim:
            return 1e9
        # 1) tempi previsti sul TRAIN
        T_pred = predict_times(w, X_train)
        # 2) errore temporale medio (paper)
        base_err = temporal_fitness_function(
            actual_spike_times=T_pred,
            target_classes=y_train,
            num_classes=num_classes,
            t_max=t_max,
            tau=tau,
        )
        # 3) regolarizzazione leggera sui pesi
        W, _ = unpack(w)
        reg = float(lambda_l2) * float(np.sum(W**2))

        fitness = float(base_err + reg)
        return fitness if np.isfinite(fitness) else 1e9

    return obj_fun, dim, predict_times


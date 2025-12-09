import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from functions.temporal_rank_order_encoding import temporal_rank_order_encode
# ------------------------------------------------------------
# Parametri SNN / codifica
# ------------------------------------------------------------
T = 256
t_min = 0
t_max = T - 1

device = "cpu"

# ------------------------------------------------------------
# 1. Carica Iris e fai train/test split
# ------------------------------------------------------------
iris = load_iris()
X = iris.data.astype(np.float32)  # [150, 4]
y = iris.target.astype(np.int64)    # [150]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ------------------------------------------------------------
# 2. Calcola min/max SOLO sul training
# ------------------------------------------------------------
x_min = X_train.min(axis=0) # [F]
x_max = X_train.max(axis=0)

train_times = temporal_rank_order_encode(X_train, x_min, x_max, T)  # [N_tr, F]
test_times  = temporal_rank_order_encode(X_test,  x_min, x_max, T)  # [N_te, F]

# ------------------------------------------------------------
# 3. Da tempi di spike a spike train binari [T, B, F]
#    (versione compatibile col tuo workflow)
# ------------------------------------------------------------
def times_to_trains(spike_times, T=T, device=device):
    """
    spike_times: numpy array [B, F] con valori interi in [0, T-1]
    ritorna:    torch.Tensor [T, B, F] con spike binari (0/1)
    """
    t = torch.as_tensor(spike_times, device=device, dtype=torch.long)  # [B, F]
    B, F = t.shape

    S = torch.zeros(T, B, F, device=device)

    # indici batch/feature
    b_ix = torch.arange(B, device=device).unsqueeze(1).expand_as(t)   # [B, F]
    f_ix = torch.arange(F, device=device).unsqueeze(0).expand_as(t)   # [B, F]

    # metti 1 nel time-step corrispondente
    S[t, b_ix, f_ix] = 1.0

    return S  # [T, B, F]


S_train = times_to_trains(train_times, T=T, device=device)  # [T, N_tr, 4]
S_test  = times_to_trains(test_times,  T=T, device=device)  # [T, N_te, 4]

# converti anche y in tensori
y_train_tensor = torch.as_tensor(y_train, device=device, dtype=torch.long)
y_test_tensor  = torch.as_tensor(y_test,  device=device, dtype=torch.long)


# train_times: [N_train, 4] (int in [0,255])
# normalizziamo un po'
X_times_tr = torch.as_tensor(train_times, dtype=torch.float32, device=device) / 255.0
X_times_te = torch.as_tensor(test_times,  dtype=torch.float32, device=device) / 255.0

y_tr = torch.as_tensor(y_train, dtype=torch.long, device=device)
y_te = torch.as_tensor(y_test,  dtype=torch.long, device=device)



import torch
import numpy as np


from models.spike_nn import SpikeNeuralNetwork
from functions.temporal_rank_order_encoding import temporal_rank_order_encode
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from functions.times_to_trains import times_to_trains

from sklearn.model_selection import train_test_split 

# ------------------------------------------------------------
# 0. Parametri SNN / codifica
# ------------------------------------------------------------
T = 20
t_min = 0
t_max = T - 1
beta = 1.0
threshold = 1.0
hidden_dim = 32
device = "cpu"
l2 = 0.001

# ------------------------------------------------------------
# 1. Import Dataset
# ------------------------------------------------------------
iris = load_iris()
X = iris.data.astype(np.float32)    # [N, F]
y = iris.target.astype(np.float32)  # [N]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ------------------------------------------------------------
# 2. Computing spiking times
# ------------------------------------------------------------
x_min = X_train.min(axis=0)  # [F]
x_max = X_train.max(axis=0)  # [F]
train_times = temporal_rank_order_encode(X_train, x_min, x_max, T)
test_times = temporal_rank_order_encode(X_test, x_min, x_max, T)

# ------------------------------------------------------------
# 3. From spike times to spike trains: [T, B, F]
# ------------------------------------------------------------
S_train = times_to_trains(train_times, T, device)
S_test = times_to_trains(test_times, T, device)

# Convert into tensor
S_train_tensor = torch.as_tensor(S_train, dtype=torch.float32, device=device)
S_test_tensor = torch.as_tensor(S_test, dtype=torch.float32, device=device)

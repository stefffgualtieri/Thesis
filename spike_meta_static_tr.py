import torch
import torch.nn as nn
import numpy as np
import time

from models.spike_nn import SpikeNeuralNetwork
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from functions.optimizers.hiking_spike_static import hiking_opt_spike_static
from functions.utils import to_static_seq

#----------------------------------------------------------------------------
# Import and Pre-processing
#----------------------------------------------------------------------------
iris = load_iris()

X = iris.data.astype(np.float32)
y_raw = iris.target

le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

#----------------------------------------------------------------------------
# Parameters and network
#----------------------------------------------------------------------------
beta = 0.95
T = 20

input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = int(torch.unique(y_train).numel())

net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=3.0,
    bias=False
)
ce = nn.CrossEntropyLoss()


#----------------------------------------------------------------------------
# Training
#----------------------------------------------------------------------------
start_time = time.time()
best_w, best_iteration = hiking_opt_spike_static(
    obj_fun=ce,
    X_train=X_train,
    y_train=y_train,
    model_snn=net,
    lower_b=-5.0,
    upper_b=5.0,
    pop_size=100,
    max_iter=100,
    seed=42,
    T=T   
)
print(f"Training time: {time.time() - start_time:.3f} s")


#-------------------------------------------------------------------------------------
# Copy the best weights obtained in the model 
#-------------------------------------------------------------------------------------

# linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
# idx = 0
# with torch.no_grad():
#     for lin in linear_layers:
#         #Weights
#         number_weights = lin.weight.numel()
#         lin.weight.copy_(best_w[idx:idx + number_weights].view_as(lin.weight))
#         idx += number_weights
#         #Bias
#         if lin.bias is not None:
#             number_bias = lin.bias.numel()
#             lin.bias.copy_(best_w[idx: idx + number_bias].view_as(lin.bias))
#             idx += number_bias

#----------------------------------------------------------------------------
# Evaluation
#----------------------------------------------------------------------------
net.eval()
with torch.no_grad():
    spike_times, _ = net(to_static_seq(X_test, T))
    spike_counts = spike_times.sum(dim=0)
    logits = spike_counts.float()
    loss = ce(logits, y_test.long()).item()
    y_pred = logits.argmax(dim=1)
    acc = (y_pred == y_test).float().mean().item()
print(f"Test Loss: {loss},\t Test Accuracy: {acc}")
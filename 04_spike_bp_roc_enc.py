import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from functions.temporal_rank_order_encoding import temporal_rank_order_encode
from functions.times_to_trains import times_to_trains

from models.spike_nn import SpikeNeuralNetwork
# ------------------------------------------------------------
# 0. Parameters
# ------------------------------------------------------------
T = 20
t_min = 0
t_max = T - 1
beta = 1.0
threshold = 0.1
hidden_dim = 32
lr = 0.001
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
    stratify=y
)

# ------------------------------------------------------------
# 2. Computing spiking times
# ------------------------------------------------------------
x_min = X_train.min(axis=0) # [F]
x_max = X_train.max(axis=0) # [F]

train_times = temporal_rank_order_encode(X_train, x_min, x_max, T)  # [N_tr, F]
test_times  = temporal_rank_order_encode(X_test,  x_min, x_max, T)  # [N_te, F]

# ------------------------------------------------------------
# 3. From spike times to spike trains: [T, B, F]
# ------------------------------------------------------------

S_train = times_to_trains(train_times, T=T, device=device)  # [T, N_tr, F]
S_test  = times_to_trains(test_times,  T=T, device=device)  # [T, N_te, F]

# Convert in torch.Tensor
S_train_tensor = torch.as_tensor(S_train, dtype=torch.float32, device=device)
S_test_tensor = torch.as_tensor(S_test, dtype=torch.float32, device=device)
y_train_tensor = torch.as_tensor(y_train, device=device, dtype=torch.long)
y_test_tensor  = torch.as_tensor(y_test,  device=device, dtype=torch.long)

# ------------------------------------------------------------
# 4.1 Setup for Training
# ------------------------------------------------------------

_, N, F = S_train_tensor.shape
num_classes = int(y_train_tensor.max().item() + 1)

net = SpikeNeuralNetwork(
    input_dim=F,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    beta=beta,
    threshold=threshold,
).to(device)

ce = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=lr)

# ------------------------------------------------------------
# 4.2 Training Loop
# ------------------------------------------------------------

num_epochs = 200

for epoch in range(1, num_epochs + 1):
    net.train()
    
    # Forward
    spk_tr, _ = net(S_train_tensor)  #[T, N, C]
    logits_tr = spk_tr.sum(dim=0)   #[N, C]
    loss_tr = ce(logits_tr, y_train_tensor)
    
    # check firing rate
    _, _, C = spk_tr.shape
    
    # Total number of spikes for each output neuron
    spikes_per_neuron = spk_tr.sum(dim=(0,1)) #[C]
    
    # Firing Rate
    firing_rate_per_neuron = spikes_per_neuron / (T*N)
    
    if epoch % 10 == 0:
        print(f"\n\n Epoch {epoch} | Firing Rate per Neuron: {firing_rate_per_neuron}")
    
    # Backward
    opt.zero_grad()
    loss_tr.backward()
    opt.step()
    
    # Acc tr
    pred_tr = logits_tr.argmax(dim=1)
    acc_tr = (pred_tr == y_train_tensor).float().mean().item()
    
    print(f"Epoch {epoch} | loss_tr {loss_tr:.4f}, acc_tr {acc_tr:.4f}"
    )
    
# Eval on test
net.eval()
with torch.no_grad():
    spk_te, _ = net(S_test_tensor)
    logits_te = spk_te.sum(dim=0)
    loss_te  = ce(logits_te, y_test_tensor)
    acc_te = (logits_te.argmax(dim=1) == y_test_tensor).float().mean().item()
    
    print(f"Test loss: {loss_te:.4f}, Test accuracy: {acc_te:.4f}")
        
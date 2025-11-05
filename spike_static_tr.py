import torch
from torch import nn
import numpy as np

from models.spike_nn import SpikeNeuralNetwork

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler
from functions.utils import to_static_seq

# 1) Import the dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features.values.astype(np.float32)
y_raw = iris.data.targets.values.ravel()

le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Parameters and network
beta = 0.95
T = 50
tau = 20.0
epochs = 500

input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 3

net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=1
)
ce = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

#Training Loop
for epoch in range(1, epochs + 1):
    net.train()

    perm = torch.randperm(X_train.size(0))
    x_tr = X_train[perm]
    y_tr = y_train[perm]

    spk_tr, _ = net(to_static_seq(x_tr, T))
    logits_tr = spk_tr.sum(dim=0)
    loss_tr = ce(logits_tr, y_tr)

    # Backward
    opt.zero_grad()
    loss_tr.backward()
    opt.step()

    #eval rapido
    with torch.no_grad():
        #train
        pred_tr = logits_tr.argmax(dim=1)
        acc_tr = (pred_tr == y_tr).float().mean().item()
        
        #test
        net.eval()
        spk_te, _ = net(to_static_seq(X_test, T))
        logits_te = spk_te.sum(dim=0)
        loss_te = ce(logits_te, y_test).item()
        acc_te = (logits_te.argmax(dim=1) == y_test).float().mean().item()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch} | loss {loss_tr.item():.4f} acc_tr {acc_tr:.3f} | "
              f"loss_test {loss_te:.4f} acc_test {acc_te:.3f}")

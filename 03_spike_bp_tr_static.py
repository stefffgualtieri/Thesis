import torch
from torch import nn
import numpy as np

from models.spike_nn import SpikeNeuralNetwork

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from functions.optimizers.utils import to_static_seq

#-------------------------------------------------------------------------------------
# Pre Processing
#-------------------------------------------------------------------------------------
dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

#-------------------------------------------------------------------------------------
# Defining the networks
#-------------------------------------------------------------------------------------
beta = 0.95
T = 20
epochs = 500
bias = False
lr = 0.1

input_dim = X_train.shape[1]
hidden_dim = 50
output_dim = int(torch.unique(y_train_tensor).numel())

model_snn = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=1,
    bias=bias
)
ce = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model_snn.parameters(), lr=lr)

# Prepare the dataset
x_tr = to_static_seq(X_train_tensor, T)
x_te = to_static_seq(X_test_tensor, T)

#-------------------------------------------------------------------------------------
# Training Loop
#-------------------------------------------------------------------------------------
model_snn.train()
for epoch in range(epochs):
    # Shuffle the training dataset
    # perm = torch.randperm(X_train_tensor.size(0))
    # x_tr = X_train_tensor[perm]
    # y_tr = y_train_tensor[perm]

    # Forward
    spk_tr, _ = model_snn(x_tr)
    logits_tr = spk_tr.sum(dim=0)
    loss_tr = ce(logits_tr, y_train_tensor)

    # Backward
    opt.zero_grad()
    loss_tr.backward()
    opt.step()

    with torch.no_grad():
        pred_tr = logits_tr.argmax(dim=1)
        acc_tr = (pred_tr == y_train_tensor).float().mean().item()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch} | loss {loss_tr.item():.4f} acc_tr {acc_tr:.3f} |")

#-------------------------------------------------------------------------------------
# Evaluation on test dataset
#-------------------------------------------------------------------------------------
with torch.no_grad():
    model_snn.eval()
    spk_te, _ = model_snn(x_te)
    logits_te = spk_te.sum(dim=0)
    loss_te = ce(logits_te, y_test_tensor).item()
    acc_te = (logits_te.argmax(dim=1) == y_test_tensor).float().mean().item()

    print(f"Loss Test: {loss_te:.4f} | Accuracy Test: {acc_te:.4f} |")
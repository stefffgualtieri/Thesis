import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.spike_nn import SpikeNeuralNetwork

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo

# Loading dataset
dataset_id = 53
dataset = fetch_ucirepo(id=dataset_id)
X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy().ravel()

# Normalization
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

le = LabelEncoder()
y_num = le.fit_transform(y)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_num, test_size=0.2, random_state=42, stratify=y_num)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# Utils
num_inputs = 4
num_outputs = 3
num_hidden = 10

num_steps = 20
beta = 1

# Define the model
net = SpikeNeuralNetwork(
    input_dim=num_inputs,
    hidden_dim=num_hidden,
    output_dim=num_outputs,
    beta=beta,
    threshold=1
)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

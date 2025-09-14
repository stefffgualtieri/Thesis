import torch
from torch import nn
import matplotlib.pyplot as plt
from models.classic_nn_no_grad import NeuralNetwork

from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions.optimizers.hiking_opt import hiking_optimization


#Load the data
dataset = load_iris()
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#Define the model
input_dim = X_train_tensor.size(dim=1)
output_dim = 3
loss_fn = nn.CrossEntropyLoss()

model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim)

best_w, best_iteration = hiking_optimization(
    obj_fun=loss_fn,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    model_snn=model,
    pop_size=100,
    max_iter=500
)

linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
'''
Copia i valori dal vettore piatto w_flat nei pesi.

Parametri:
    layers : lista di layer nn.Linear
    w_flat : torch.Tensor 1D contenente tutti i parametri concatenati
'''
idx = 0
with torch.no_grad():
    for lin in linear_layers:
        #Weights
        number_weights = lin.weight.numel()
        lin.weight.copy_(best_w[idx:idx + number_weights].view_as(lin.weight))
        idx += number_weights
        #Bias
        if lin.bias is not None:
            number_bias = lin.bias.numel()
            lin.bias.copy_(best_w[idx: idx + number_bias].view_as(lin.bias))
            idx += number_bias

#Evaluation
model.eval()

with torch.no_grad():
    logits = model(X_test_tensor)
    loss = loss_fn(logits, y_test_tensor).item()
    y_pred = logits.argmax(dim=1)
    acc = (y_pred == y_test_tensor).float().mean().item()
print(f"Test Loss: {loss},\t Test Accuracy: {acc}")
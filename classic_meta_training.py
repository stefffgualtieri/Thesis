import torch
from torch import nn
from typing import List

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.classic_nn import NeuralNetwork
from functions.optimizers.hiking_opt import hiking_optimization

@torch.no_grad()
def assign_from_flat_layers(layers: List[nn.Linear], w_flat: torch.Tensor) -> None:
    """
    Copia i valori dal vettore piatto w_flat nei pesi e bias dei layer Linear.
    
    Parametri:
        layers : lista di layer nn.Linear
        w_flat : torch.Tensor 1D contenente tutti i parametri concatenati
    """
    idx = 0
    for lin in layers:
        # pesi
        n_w = lin.weight.numel()
        lin.weight.copy_(w_flat[idx:idx+n_w].view_as(lin.weight))
        idx += n_w
        # bias (se presente)
        if lin.bias is not None:
            n_b = lin.bias.numel()
            lin.bias.copy_(w_flat[idx:idx+n_b].view_as(lin.bias))
            idx += n_b
    assert idx == w_flat.numel(), "Dimensioni incoerenti: w_flat troppo corto/lungo"


#Collect Data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

loss_fn = nn.CrossEntropyLoss()

model = NeuralNetwork()

best_w, best_iteration = hiking_optimization(
    obj_fun=loss_fn,
    X_train = X_train_tensor,
    y_train = y_train_tensor,
    model_snn=model,
    pop_size=200,
    max_iter=500
)

layers = [model.linear_relu_stack[0], model.linear_relu_stack[2]]

assign_from_flat_layers(layers, best_w)

model.eval()
logits = model(X_test_tensor)
loss = loss_fn(logits, y_test_tensor).item()
preds = logits.argmax(dim=1)
acc = (preds == y_test_tensor).float().mean().item()

print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc*100:.2f}%")
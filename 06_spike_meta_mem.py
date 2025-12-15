import torch
import torch.nn as nn
import pygmo as pg
import numpy as np

from functions.optimizers.utils import vector_to_weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from functions.optimizers.utils import to_static_seq, get_linear_layers, dim_from_layers
from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_CE_mem import SNNProblem_CE_mem
#----------------------------------------------
# Pre-Processing
#----------------------------------------------
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


T = 20
input_dim = X_train.shape[1]
hidden_dim = 16
num_classes = int(torch.unique(y_train_tensor).numel())

beta = 1.0
threshold = 0.5
bias = False

X_train_tensor = to_static_seq(X_train_tensor, T)
X_te_tensor = to_static_seq(X_test_tensor, T)

#----------------------------------------------
# Defining the model and the problem
#----------------------------------------------
device = "cpu"
net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    beta=beta,
    threshold=threshold,
    bias=bias
).to(device)

layers = get_linear_layers(net)
dim = dim_from_layers(layers)
lb = [-3.0] * dim
ub = [3.0] * dim

prob = pg.problem(SNNProblem_CE_mem(
    model=net,
    X=X_train_tensor, 
    y=y_train_tensor,
    layers=layers,
    lb=lb,
    ub=ub,
    device=device
))
algo = pg.algorithm(pg.pso(
    gen=100
))
algo.set_verbosity(1)
pop = pg.population(prob, size=50, seed=42)

pop = algo.evolve(pop)


best_w = pop.champion_x
best_f = pop.champion_f[0]
print(f"Best Train CE: {best_f}")

#-------------------------------
# Evaluation for the best hiker
#-------------------------------
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device=device)
vector_to_weights(best_w_t, layers)

net.eval()
with torch.no_grad():
    spk_te, mem_te = net(X_te_tensor)
    logits_te = mem_te.mean(dim=0)
    pred_te = logits_te.argmax(dim=1)
    acc_te = (pred_te == y_test_tensor).float().mean().item()
    loss_te = torch.nn.CrossEntropyLoss()(logits_te, y_test_tensor).item()

print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")
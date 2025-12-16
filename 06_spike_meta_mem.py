import torch
import torch.nn as nn
import pygmo as pg
import numpy as np
import pandas as pd
import json
from pathlib import Path

from functions.utils.utils import vector_to_weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from functions.utils.utils import to_static_seq, get_linear_layers, dim_from_layers
from functions.utils.utils_memmean_metrics import spike_rate_mean, spike_count_per_sample, spike_rate_per_class
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

# Parameters
T = 20
input_dim = X_train.shape[1]
hidden_dim = 32
num_classes = int(torch.unique(y_train_tensor).numel())

beta = 1.0
threshold = 0.5
bias = False
ce = nn.CrossEntropyLoss()
gen = 100
pop = 50

X_train_tensor = to_static_seq(X_train_tensor, T)
X_test_tensor = to_static_seq(X_test_tensor, T)

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

upd = SNNProblem_CE_mem(
    model=net,
    X=X_train_tensor, 
    y=y_train_tensor,
    layers=layers,
    lb=lb,
    ub=ub,
    device=device,
    ce = ce,
)

#-------------------------------------------
# Training Loop
#-------------------------------------------
prob = pg.problem(upd)

algo = pg.algorithm(pg.pso(
    gen=gen
))
algo.set_verbosity(1)
pop = pg.population(prob, size=pop, seed=42)
pop = algo.evolve(pop)
#-------------------------------------------
# log = algo.extract(pg.pso).get_log()
# df = pd.DataFrame(log, columns=[
#     "Gen",
#     "fevals",
#     "gbest",
#     "mean_vel",
#     "mean_lbest",
#     "avg_dist"
# ])

# dir= "results/pso"
# Path(dir).mkdir(parents=True, exist_ok=True)

# df.to_csv(f"{dir}/pso_iris_memmean_log.csv", index=False)



#-------------------------------
# Evaluation for the best hiker
#-------------------------------
best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device=device)
vector_to_weights(best_w_t, layers)
net.eval()
with torch.no_grad():
    # Train
    spk_tr, mem_tr = net(X_train_tensor)
    logits_tr = mem_tr.mean(dim=0)
    pred_tr = logits_tr.argmax(dim=1)
    acc_tr = (pred_tr == y_train_tensor).float().mean().item()
    loss_tr = ce(logits_tr, y_train_tensor).item()
    
    # Test
    spk_te, mem_te = net(X_test_tensor)
    logits_te = mem_te.mean(dim=0)
    pred_te = logits_te.argmax(dim=1)
    acc_te = (pred_te == y_test_tensor).float().mean().item()
    loss_te = ce(logits_te, y_test_tensor).item()

    sr_mean = spike_rate_mean(spk_te)
    sr_cls = spike_rate_per_class(spk_te).cpu().tolist()
    spikes_per_sample = spike_count_per_sample(spk_te)
print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")

# summary = {
#     "algorithm": "PSO",
#     "dataset": "iris",

#     "train_loss": float(loss_tr),
#     "train_acc": float(acc_tr),
#     "test_loss":  float(loss_te),
#     "test_acc":   float(acc_te),
#     "spike_rate_mean_test": float(sr_mean),
#     "spike_rate_per_class_test": sr_cls,
#     "spike_per_sample_test": spikes_per_sample,
# }
# with open(f"{dir}/pso_iris_summary.json", "w") as f:
#     json.dump(summary, f, indent=2)
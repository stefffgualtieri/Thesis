import torch
import numpy as np
import pygmo as pg

from functions.SNNProblem_CE import SNNProblem_CE

from models.spike_nn import SpikeNeuralNetwork
from functions.temporal_rank_order_encoding import temporal_rank_order_encode
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from functions.times_to_trains import times_to_trains
from functions.utils.utils import get_linear_layers, dim_from_layers

from functions.utils.utils import vector_to_weights

from sklearn.model_selection import train_test_split 

# ------------------------------------------------------------
# 0. Parametri SNN / codifica
# ------------------------------------------------------------
T = 20
device = "cpu"


# ------------------------------------------------------------
# 1. Import Dataset
# ------------------------------------------------------------
iris = load_iris()
X = iris.data.astype(np.float32)    # [N, F]
y = iris.target.astype(np.int64)  # [N]

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

y_train_tensor = torch.as_tensor(y_train, dtype=torch.long, device=device)
y_test_tensor  = torch.as_tensor(y_test, dtype=torch.long, device=device)


#------------------------------------------------------------
# Creating the model
#-----------------------------------------------------------

_, N, F = S_train_tensor.shape
num_classes = int(y_train_tensor.max().item() + 1)

bias = False
beta = 1.0
threshold = 1.0
hidden_dim = 16

net = SpikeNeuralNetwork(
    input_dim=F,
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

prob = pg.problem(SNNProblem_CE(
    model=net,
    X=S_train_tensor,
    y=y_train_tensor,
    layers=layers,
    lb=lb,
    ub=ub
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
    spk_te, _ = net(S_test_tensor)
    logits_te = spk_te.sum(dim=0)
    pred_te = logits_te.argmax(dim=1)
    acc_te = (pred_te == y_test_tensor).float().mean().item()
    loss_te = torch.nn.CrossEntropyLoss()(logits_te, y_test_tensor).item()

print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")
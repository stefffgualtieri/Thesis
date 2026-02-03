import torch
import numpy as np
import pygmo as pg

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from functions.utils.utils import first_spike_times, vector_to_weights, temporal_encoding, times_to_trains, get_linear_layers, dim_from_layers
from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_te import SNNProblem_te
from functions.temporal_fitness import temporal_fitness_function

#----------------------------------------------
# 1.1 Pre-Processing
#----------------------------------------------
seed = 42

iris = load_iris()
X = iris.data.astype(np.float32)    # [N, F]
y = iris.target.astype(np.int64)    # [N]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state=seed, 
    stratify=y,
    test_size=0.4
)

# from np.array to torch.Tensor
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Min e max of train test
x_min = X_train.min(dim=0).values   # [F]
x_max = X_train.max(dim=0).values   # [F]


T = 100
X_train_t = temporal_encoding(X_train, x_min, x_max, T)     # [N, F]
X_test_t = temporal_encoding(X_test, x_min, x_max, T)       # [N, F]

S_train = times_to_trains(X_train_t, T)     # [T, N, F]
S_test = times_to_trains(X_test_t, T)       # [T, N, F]

#----------------------------------------------
# 1.2 Defining the model
#----------------------------------------------
input_dim = X_train.shape[1]
hidden_dim = 64
num_classes = int(torch.unique(y_train).numel())
beta = 1.0
threshold = 1.0
bias = False
device = "cpu"

# Model
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
print(f"Dimension: {dim}")
lb = [-5.0] * dim
ub = [5.0] * dim
tau = 20

upd = SNNProblem_te(
    model=net,
    X=S_train,
    y=y_train,
    layers=layers,
    lb=lb,
    ub=ub,
    loss_f=temporal_fitness_function,
    tau=tau,
    T=T,
    device=device
)

#----------------------------------------------
# 2 Training Loop
#----------------------------------------------
gen = 50
pop = 30

prob = pg.problem(upd)
algo = pg.algorithm(pg.pso(
    gen=gen,
    seed=seed
))
algo.set_verbosity(1)
pop = pg.population(prob, size=pop, seed=seed)
pop = algo.evolve(pop)

#----------------------------------------------
# 3 Evaluation
#----------------------------------------------
best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device=device)
vector_to_weights(best_w_t, layers)

net.eval()
with torch.no_grad():
    # Train
    spk_tr, mem = net(S_train)
    t_first_tr = first_spike_times(spk_tr)
    loss_tr = temporal_fitness_function(
        t_first_tr,
        y_train,
        T,
        tau
    )
    y_pred_tr = t_first_tr.argmin(dim=1)    # [B]
    acc_tr = (y_pred_tr == y_train).float().mean().item()

    no_spike_rate = (t_first_tr >= T).float().mean(dim=0)
    mask = t_first_tr < T
    mean_fired = (t_first_tr.float() * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
    print(f"Mean t_first | fired only: {mean_fired}")
    print(f"No-spike rate per neurone: {no_spike_rate}")
    print(f"Mean t_first: {t_first_tr.float().mean(dim=0)}")

    print(f"Train loss: {loss_tr:.4f} | train acc: {acc_tr:.4f}")
    fired_any = (t_first_tr < T).any(dim=1).float().mean()
    print("Fraction esempi con almeno 1 spike in output:", fired_any.item())

    print(f"mem max: {mem.max().item()}")
    print(f"mem 99%: {torch.quantile(mem.flatten(), 0.99).item()}")

    # Test
    spk_te, _ = net(S_test)
    t_first_te = first_spike_times(spk_te)
    loss_te = temporal_fitness_function(
        t_first_te,
        y_test,
        T,
        tau
    )
    y_pred_te = t_first_te.argmin(dim=1)    # [B]
    acc_te = (y_pred_te == y_test).float().mean().item()

    print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")
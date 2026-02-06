from functions.load_adult import load_adult
import torch
import torch.nn as nn
import pygmo as pg

from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_CE_mem import SNNProblem_CE_mem

from functions.utils.utils import dim_from_layers, get_linear_layers, vector_to_weights

#--------------------------
# Load Dataset
#--------------------------
X_train, X_test, y_train, y_test = load_adult()

# Sanity Check
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
print("Train positive rate:", y_train.float().mean().item())
print("Test  positive rate:", y_test.float().mean().item())

#parameters
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 2
bias = False

beta = 0.9
threshold = 1.0
num_steps = 1

gen = 150
pop = 60

# Defining the newtork
net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=threshold,
    bias=bias
)

layers = get_linear_layers(net)
dim = dim_from_layers(layers)
print(dim)
lb = [-3] * dim
ub = [3] * dim
ce = nn.CrossEntropyLoss()

upd = SNNProblem_CE_mem(
    model=net,
    X=X_train,
    y=y_train,
    layers=layers,
    lb=lb,
    ub=ub,
    ce=ce,
    num_steps=num_steps,
    device="cpu"
)

prob = pg.problem(upd)
algo = pg.algorithm(pg.gwo(
    gen=gen
))
algo.set_verbosity(1)
pop = pg.population(prob, size=pop, seed=42)
pop = algo.evolve(pop)

best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device="cpu")
vector_to_weights(best_w_t, layers)
net.eval()
with torch.no_grad():
    # Train
    spk_tr, mem_tr = net(X_train, num_steps)
    logits_tr = mem_tr.mean(dim=0)
    pred_tr = logits_tr.argmax(dim=1)
    acc_tr = (pred_tr == y_train).float().mean().item()
    loss_tr = ce(logits_tr, y_train).item()
    print(logits_tr.abs().max().item(), logits_tr.abs().mean().item())

    
    # Test
    spk_te, mem_te = net(X_test, num_steps)
    logits_te = mem_te.mean(dim=0)
    pred_te = logits_te.argmax(dim=1)
    acc_te = (pred_te == y_test).float().mean().item()
    loss_te = ce(logits_te, y_test).item()
    print(logits_te.abs().max().item(), logits_te.abs().mean().item())

    
    print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")


#Resul: Test loss: 5.2917 | Test acc: 0.8700
#Result: Test loss: 0.3772 | Test acc: 0.8500
#Result: Test loss: 0.4707 | Test acc: 0.8400
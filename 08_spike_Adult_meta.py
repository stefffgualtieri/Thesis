from functions.load_adult import load_adult
from functions.load_adult_balanced import load_adult_balanced
import torch
import torch.nn as nn
import pygmo as pg

from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_snn import SNNProblem_snn

from functions.utils.utils import dim_from_layers, get_linear_layers, vector_to_weights
from functions.metrics import precision_recall_f1_binary

#--------------------------
# Load Dataset
#--------------------------
X_train, X_test, y_train, y_test = load_adult_balanced()

# Sanity Check
# print("Train:", X_train.shape, y_train.shape)
# print("Test :", X_test.shape, y_test.shape)
# print("Train positive rate:", y_train.float().mean().item())
# print("Test  positive rate:", y_test.float().mean().item())

#parameters
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 2
bias = False

beta = 0.9
threshold = 1.0
T = 20

gen = 50
pop = 70

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
lb = [-1] * dim
ub = [1] * dim
ce = nn.CrossEntropyLoss()

upd = SNNProblem_snn(
    model=net,
    X=X_train,
    y=y_train,
    layers=layers,
    lb=lb,
    ub=ub,
    ce=ce,
    num_steps=T,
    device="cpu"
)

prob = pg.problem(upd)
algo = pg.algorithm(pg.sga(
    gen=gen
))
algo.set_verbosity(1)
pop = pg.population(prob, size=pop, seed=42)
pop = algo.evolve(pop)


# Evaluation
best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device="cpu")
vector_to_weights(best_w_t, layers)
net.eval()
with torch.no_grad():
    # Train
    spk_tr, mem_tr, _ = net(X_train, T)
    logits_tr = mem_tr.mean(dim=0)
    pred_tr = logits_tr.argmax(dim=1)
    acc_tr = (pred_tr == y_train).float().mean().item()
    loss_tr = ce(logits_tr, y_train).item()
    print(f"Train loss: {loss_tr:.4f} | Train acc: {acc_tr:.4f}")

    
    # Test
    spk_te, mem_te, energy_te = net(X_test, T)
    logits_te = mem_te.mean(dim=0)
    pred_te = logits_te.argmax(dim=1)
    acc_te = (pred_te == y_test).float().mean().item()
    loss_te = ce(logits_te, y_test).item()

    
    print(f"Test loss: {loss_te:.4f} | Test acc: {acc_te:.4f}")

    p, r, f1 = precision_recall_f1_binary(pred_te, y_test)


out_dir = "results/adult/snn/sga"

print(f"Evaluation on the test set:")
print(f"Test Loss: {loss_te}")
print(f"Test Acc: {acc_te}")
print(f"Test precision: {p}")
print(f"Test recall: {r}")
print(f"Test f1-score: {f1}")
print(f"Test Energy: {energy_te:.4f}")

with open(out_dir + "/adult_snn_sga_3.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation on the test set\n")
    f.write(f"Test Loss: {loss_te:.5f}\n")
    f.write(f"Test Acc: {acc_te:.5f}\n")
    f.write(f"Test Precision: {p:.5f}\n")
    f.write(f"Test Recall: {r:.5f}\n")
    f.write(f"Test f1: {f1:.5f}\n")
    f.write(f"Test Energy per sample: {(energy_te / T):.5f}")


#Resul: Test loss: 5.2917 | Test acc: 0.8700
#Result: Test loss: 0.3772 | Test acc: 0.8500
#Result: Test loss: 0.4707 | Test acc: 0.8400
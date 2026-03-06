import torch
import torch.nn as nn
import pygmo as pg
import numpy as np

from functions.metrics import precision_recall_f1_binary, macro_precision_recall_f1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from functions.utils.utils import get_linear_layers, dim_from_layers, vector_to_weights
from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_snn import SNNProblem_snn

#----------------------------------------------
# Pre-Processing
#----------------------------------------------
random_state = 42

iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

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

beta = 0.95
threshold = 0.5
bias = False

ce = nn.CrossEntropyLoss()

gen = 40
pop = 30

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
print(f"Dimension: {dim}")
lb = [-2.0] * dim
ub = [2.0] * dim

upd = SNNProblem_snn(
    model=net,
    X=X_train_tensor, 
    y=y_train_tensor,
    layers=layers,
    lb=lb,
    ub=ub,
    device=device,
    ce = ce,
    num_steps=T
)

#-------------------------------------------
# Training Loop
#-------------------------------------------
prob = pg.problem(upd)

algo = pg.algorithm(pg.gaco(
    gen=gen,
    ker=pop
))
algo.set_verbosity(1)   #for visualization
pop = pg.population(prob, size=pop, seed=random_state)
pop = algo.evolve(pop)

#-------------------------------
# Evaluation for the best hiker
#-------------------------------
best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device=device)
vector_to_weights(best_w_t, layers)

net.eval()
with torch.no_grad():
    # Train
    spk_tr, mem_tr, _ = net(X_train_tensor, T)
    logits_tr = mem_tr.mean(dim=0)
    pred_tr = logits_tr.argmax(dim=1)
    
    acc_tr = (pred_tr == y_train_tensor).float().mean().item()
    loss_tr = ce(logits_tr, y_train_tensor).item()
    
    # Test
    spk_te, mem_te, energy_te = net(X_test_tensor, T)
    logits_te = mem_te.mean(dim=0)
    test_pred = logits_te.argmax(dim=1)
    
    test_acc = (test_pred == y_test_tensor).float().mean().item()
    test_loss = ce(logits_te, y_test_tensor).item()
    
    p, r, f1 = macro_precision_recall_f1(test_pred, y_test_tensor, num_classes)
    energy_te = energy_te.item()
    
    print(f"Acc Train: {acc_tr} | Acc Test: {test_acc}")
    
print(f"Evaluation on the test set:")
print(f"Test Loss: {test_loss}")
print(f"Test Acc: {test_acc}")
print(f"Test precision: {p}")
print(f"Test recall: {r}")
print(f"Test f1-score: {f1}")
print(f"Test Energy per sample: {(energy_te / T):.5f}")

out_dir = 'results/iris/snn/gaco'

with open(out_dir + "/iris_snn_gaco_2.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation on the test set\n")
    f.write(f"Test Loss: {test_loss:.5f}\n")
    f.write(f"Test Acc: {test_acc:.5f}\n")
    f.write(f"Test Precision: {p:.5f}\n")
    f.write(f"Test Recall: {r:.5f}\n")
    f.write(f"Test f1: {f1:.5f}\n")
    f.write(f"Test Energy per sample: {(energy_te / T):.5f}")
import torch
from torch import nn
import pygmo as pg

from models.classic_nn import NeuralNetwork
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from functions.metrics import macro_precision_recall_f1, precision_recall_f1_binary
from functions.SNNProblem_CL import SNNProblem_CL
from functions.utils.utils import dim_from_layers, get_linear_layers, vector_to_weights
from functions.metrics import macro_precision_recall_f1, precision_recall_f1_binary

#-------------------------------------------------------------------------------------
# Prepare and scale the data, then convert into tensor
#-------------------------------------------------------------------------------------
random_state = 42
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#-------------------------------------------------------------------------------------
# Inizialize the model and the problem
#-------------------------------------------------------------------------------------

input_dim = X_train_tensor.size(dim=1)
hidden_dim = 32
output_dim = int(y.max().item() + 1)
bias = True

net = NeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    bias=bias
)

loss_fn = nn.CrossEntropyLoss()

layers = get_linear_layers(net)
dim = dim_from_layers(layers)

lb = [-1] * dim
ub = [1] * dim

upd = SNNProblem_CL(
    model=net,
    X=X_train_tensor,
    y=y_train_tensor,
    layers=layers,
    lb=lb,
    ub=ub,
    loss=loss_fn
)

gen = 100
pop = 30

#-------------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------------

prob = pg.problem(upd)
algo = pg.algorithm(pg.gaco(
    gen=gen,
    ker=30
))
algo.set_verbosity(1)   #for visualization
pop = pg.population(prob, size=pop, seed=random_state)

pop = algo.evolve(pop)

#-------------------------------
# Evaluation for the best hiker
#-------------------------------
best_w = pop.champion_x
best_w_t = torch.as_tensor(best_w, dtype=torch.float32, device="cpu")
vector_to_weights(best_w_t, layers)
net.eval()
with torch.no_grad(): 
    test_logits = net(X_test_tensor)
    test_loss = loss_fn(test_logits, y_test_tensor).item()

    # Prediction
    test_pred = torch.argmax(test_logits, dim=1)
    test_correct = (test_pred == y_test_tensor).sum().item()
    test_acc = test_correct/len(y_test_tensor)
    p, r, f1 = macro_precision_recall_f1(test_pred, y_test_tensor, output_dim)

print(f"Evaluation on the test set:")
print(f"Test Loss: {test_loss}")
print(f"Test Acc: {test_acc}")
print(f"Test precision: {p}")
print(f"Test recall: {r}")
print(f"Test f1-score: {f1}")

out_dir = 'results/breast_cancer'
with open(out_dir + "/breast_classic_gaco.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation on the test set\n")
    f.write(f"Test Loss: {test_loss:.5f}\n")
    f.write(f"Test Acc: {test_acc:.5f}\n")
    f.write(f"Test Precision: {p:.5f}\n")
    f.write(f"Test Recall: {r:.5f}\n")
    f.write(f"Test f1: {f1:.5f}\n")
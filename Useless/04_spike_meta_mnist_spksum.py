import torch
import snntorch as snn
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg

from models.spike_nn import SpikeConvNN
from functions.utils.utils import dim_from_layers, get_linear_layers
from functions.new_utils import balanced_indices

from functions.SNNProblem_MNIST import SNNProblem_MNIST

#----------------------------------------------
# Config
#----------------------------------------------
seed = 42
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "/data/mnist"

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download the dataset
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# -------------------------
# Build balanced subsets
# -------------------------
train_idx = balanced_indices(mnist_train, per_class=40, seed=seed)  # 40*10=400
test_idx  = balanced_indices(mnist_test,  per_class=10, seed=seed)  # 10*10=100

train_subset = Subset(mnist_train, train_idx)
test_subset  = Subset(mnist_test,  test_idx)

# -------------------------
# Convert subsets -> full tensors
# -------------------------
def subset_to_tensors(subset, device, dtype=torch.float32):
    # batch_size = len(subset) => un solo batch con tutto dentro
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    X, y = next(iter(loader))
    return X.to(device=device, dtype=dtype), y.to(device=device, dtype=torch.long)

X_train, y_train = subset_to_tensors(train_subset, device=device, dtype=dtype)
X_test,  y_test  = subset_to_tensors(test_subset,  device=device, dtype=dtype)

print("X_train:", X_train.shape, X_train.dtype, X_train.device)
print("y_train:", y_train.shape, y_train.dtype, y_train.device)
print("X_test :", X_test.shape,  X_test.dtype,  X_test.device)
print("y_test :", y_test.shape,  y_test.dtype,  y_test.device)


#----------------------------------------------
# Other Data
#----------------------------------------------
num_input = 1
num_output = 10

num_steps = 25
beta = 1.0

pop = 20
gen = 50

loss = nn.CrossEntropyLoss()

convnet = SpikeConvNN(
    beta=beta,
    input_dim=num_input,
    output_dim=num_output,
    num_steps=num_steps
)

layers = get_linear_layers(convnet)
dim = dim_from_layers(layers)
print(f"Dimension: {dim}")
lb = [-3.0] * dim
ub = [3.0] * dim

upd = SNNProblem_MNIST(
    model=convnet,
    X=X_train,
    y=y_train,
    layers=layers,
    lb=lb,
    ub=ub,
    ce=loss
)
prob = pg.problem(upd)
algo = pg.algorithm(pg.pso(
    gen=gen
))
algo.set_verbosity(1)
pop = pg.population(prob, size=pop, seed=42)
pop = algo.evolve(pop)

print("Done")



























'''
# prepara batch_cache iniziale
it = iter(train_loader)
batch_list = []
for _ in range(10):
    data, target = next(it)
    batch_list.append((data.to(device), target.to(device)))

upd.set_current_batches(batch_list)

prob = pg.problem(upd)

algo = pg.algorithm(pg.pso(
    gen=gen
))
algo.set_verbosity(1)   #for visualization
pop = pg.population(prob, size=pop, seed=42)
#-------------------------------------------
# Training Loop
#-------------------------------------------
it = iter(train_loader)

for i in range(100):
    # prepara K batch (uguali per tutte le particelle di questa generazione)
    batch_list = []
    for _ in range(K):
        try:
            data, target = next(it)
        except StopIteration:
            it = iter(train_loader)
            data, target = next(it)

        batch_list.append((
            data.to(device, non_blocking=True),
            target.to(device, non_blocking=True)
        ))

    prob.extract(upd).set_current_batches(batch_list)
    pop = algo.evolve(pop)
    
    print(f"Gen {i}: best_loss = {pop.champion_f[0]:.4f}")
'''
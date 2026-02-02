import torch
import snntorch as snn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg

from models.spike_nn import SpikeConvNN
from functions.utils.utils import dim_from_layers, get_linear_layers

from functions.SNNProblem_MNIST import SNNProblem_MNIST

#----------------------------------------------
# Pre-Processing
#----------------------------------------------
batch_size = 128
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "/data/mnist"

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))]
)

# Download the dataset
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Other Data
num_input = 1
num_output = 10
num_steps = 25
beta = 0.95
pop = 30
gen = 1
K = 10

loss = nn.CrossEntropyLoss()

convnet = SpikeConvNN(
    beta=beta,
    input_dim=num_input,
    output_dim=num_output,
    num_steps=num_steps
)

num_epochs = 100
layers = get_linear_layers(convnet)
dim = dim_from_layers(layers)
print(f"Dimension: {dim}")
lb = [-3.0] * dim
ub = [3.0] * dim

upd = SNNProblem_MNIST(
    model=convnet,
    train_loader=train_loader,
    layers=layers,
    lb=lb,
    ub=ub,
    ce=loss
)

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

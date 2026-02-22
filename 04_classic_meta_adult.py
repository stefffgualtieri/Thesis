import torch
from torch import nn
from models.classic_nn import NeuralNetwork
import matplotlib.pyplot as plt

from functions.metrics import precision_recall_f1_binary
from functions.load_adult import load_adult

from functions.utils.utils import get_linear_layers, dim_from_layers
from models.spike_nn import SpikeNeuralNetwork
from functions.SNNProblem_CL import SNNProblem_CL

#-------------------------------------------------------------------------------------
# Import dataset
#-------------------------------------------------------------------------------------
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_adult()

#-------------------------------------------------------------------------------------
# Prepare the model
#-------------------------------------------------------------------------------------
input_dim = X_train_tensor.size(dim=1)
hidden_dim = 32
output_dim = int(torch.unique(y_train_tensor).numel())
bias = True

net = SpikeNeuralNetwork(
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
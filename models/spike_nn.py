import torch
from torch import nn
import snntorch as snn

class SpikeNeuralNetwork(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim=10, output_dim=3, beta=1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta)

    #def forward(self, x):

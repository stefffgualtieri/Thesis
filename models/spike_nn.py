import torch
from torch import nn
import snntorch as snn

class SpikeNeuralNetwork(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim=10, output_dim=3, beta=1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, threshold=0.8)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.8)

    def forward(self, x, mem1, spk1, mem2):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.lif2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        return mem1, spk1, mem2, spk2
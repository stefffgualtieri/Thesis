import torch
from torch import nn
import snntorch as snn

class SpikeNeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3, beta=0.95, threshold=1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)

    def forward(self, x):
        # Initialization
        T, _, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        # training-loop
        for step in range(T):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # store in lists
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
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

    def forward(self, x, num_steps):
        # Initialization
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        # Training-loop
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Store in lists
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
    
class SpikeConvNN(nn.Module):
    def __init__(self, beta=0.95, threshold=1.0, input_dim=1, output_dim=10, num_steps=25): 
        super().__init__()
        
        # First layer
        self.conv1 = nn.Conv2d(input_dim, 6, 5, padding="same")
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.mp1 = nn.MaxPool2d(2)
        
        # Second Layer
        self.conv2 = nn.Conv2d(6, 10, 5, padding="same")
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.mp2 = nn.MaxPool2d(2)
        
        # Third Layer
        self.fc = nn.Linear(10 * 7 * 7, output_dim)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold)
        
        self.num_steps = num_steps
        
    def forward(self, x):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []
        
        # Loop
        for step in range(self.num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(cur1), mem1)
            
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(cur2), mem2)
            
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            # Store in lists
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)
    
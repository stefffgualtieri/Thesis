'''
Define a simple Neural Network of 3 layers
'''

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4,  num_classes=3):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 10, bias=False),
            nn.ReLU(),
            #nn.Linear(64, 16, bias=False),
            #nn.Dropout(0.2),
            #nn.ReLU(),
            nn.Linear(10, num_classes, bias=False),
        )
    
    @torch.no_grad()
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #probs = torch.softmax(logits, dim=1)
        return logits
import torch
from torch import nn

'''
A classic neural network to train with back_prop
Standard values are set for iris dataset
'''
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, output_dim=3):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
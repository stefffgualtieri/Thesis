from torch import nn

# A classic neural network with 3 layers (Standard values are set for iris dataset)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, output_dim=3, bias=True):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
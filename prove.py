from models.spike_nn import SpikeNeuralNetwork
from functions.optimizers.utils import get_linear_layers, dim_from_layers

net = SpikeNeuralNetwork(
    input_dim=4,
    hidden_dim=10,
    output_dim=3,
    beta=0.95,
    threshold=1,
    bias=False
)
layers = get_linear_layers(net)
print(layers)
dim = dim_from_layers(layers)
print(dim)
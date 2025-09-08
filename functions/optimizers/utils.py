import torch
import torch.nn as nn


#Copy the values in w_flat (weights + bias) in the model
@torch.no_grad()
def vector_to_weights(w_flat:torch.Tensor, layers):
    
    i = 0
    for lin in layers:
        n = lin.weight.numel()
        lin.weight.copy_(w_flat[i:i+n].view_as(lin.weight))
        i += n
        # n = lin.bias.numel()
        # lin.bias.copy_(w_flat[i:i+n].view_as(lin.bias))
        # i += n



#Function to calculate the spike times from an input and a neural network:
@torch.no_grad()
def forward_to_spike_times(model_snn: nn.Module, X: torch.Tensor, device="cpu"):
    model_snn.eval().to(device)
    X = X.to(device)

    spike_times = model_snn(X)
    spike_times = spike_times.to(torch.float32)

    return spike_times



#Function used to return a list of the linear layers of a network
def get_linear_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]



#Function to return the dimension 
def dim_from_layers(layers):
    return sum(l.weight.numel() for l in layers)
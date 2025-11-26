import torch
import torch.nn as nn


#Copy the values in w_flat (weights + bias) in the model
@torch.no_grad()
def vector_to_weights(w_flat:torch.Tensor, layers):
    i = 0
    # Iteration through the layers
    for l in layers:
        # -- Copy the weights --
        n = l.weight.numel()    # number of weights
        l.weight.copy_(w_flat[i:i+n].view_as(l.weight))
        i += n
        # -- Copy the biase --
        if l.bias is not None:
            n = l.bias.numel() # number of biases
            l.bias.copy_(w_flat[i:i+n].view_as(l.bias))
            i += n

@torch.inference_mode()
def forward_to_spike(model, x, t_sim: int = 256):
    """
    x: [T, B, F]
    return: [B, C] tempi del primo spike per classe; t_sim se nessuno spike.
    """
    spk, _ = model(x)              # spk: [T, B, C], binario/float >0 quando spike
    T, B, C = spk.shape

    fired = spk > 0                # bool [T,B,C]
    idx_first = torch.argmax(fired.float(), dim=0)   # [B,C] indice primo True (0 se nessuno)
    none_fired = ~fired.any(dim=0)           # [B,C]

    # mappa indice step -> tempo simulato
    scale = float(t_sim) / float(T)
    spike_times = idx_first.to(spk.dtype) * scale  # [B,C]
    spike_times[none_fired] = t_sim

    return spike_times

#Function used to return a list of the linear layers of a network
def get_linear_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]



#Function to return the dimension 
def dim_from_layers(layers):
    total = 0
    for l in layers:
        total += l.weight.numel()
        if l.bias is not None:
            total += l.bias.numel()
    return total

def to_static_seq(x_batch, T):
    return x_batch.unsqueeze(0).expand(T, -1, -1)/10
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

# Return a list of the linear layers of a network
def get_linear_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]



# Return the number of weights and biases
def dim_from_layers(layers):
    total = 0
    for l in layers:
        total += l.weight.numel()
        if l.bias is not None:
            total += l.bias.numel()
    return total

# Static Training
def to_static_seq(x_batch, T):
    return x_batch.unsqueeze(0).expand(T, -1, -1)/10


def times_to_trains(X_times, T=257):
    '''
    - X_times: [B, F] with values between [0, T - 1]
    - S: [T, B, F]
    '''

    t = X_times.round().clamp_(0, T-1).long()
    t = (T - 1) - t
    B, F = t.shape

    S = torch.zeros(T,B,F, device= X_times.device)
    b_ix = torch.arange(B, device=X_times.device).unsqueeze(1).expand_as(t)
    f_ix = torch.arange(F, device=X_times.device).unsqueeze(0).expand_as(t)
    S[t, b_ix, f_ix] = 1.0
    return S
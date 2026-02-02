import torch
import torch.nn as nn
import numpy as np

# Copy the values in w_flat (weights + bias) in the model
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
    return [m for m in model.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)]


# Return the number of weights and biases
def dim_from_layers(layers):
    total = 0
    for l in layers:
        total += l.weight.numel()
        if l.bias is not None:
            total += l.bias.numel()
    return total

# Input: [B, D], Output: [T, B, D]
def to_static_seq(x_batch, T, scale=0.1):
    return x_batch.unsqueeze(0).expand(T, -1, -1) * scale


def times_to_trains_alt(X_times, T=257):
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

def first_spike_times(spk_TBC: torch.Tensor) -> torch.Tensor:
    """
    spk_TBC: [T, B, C] valori 0/1
    Ritorna: [B, C] con il tempo del primo spike (int).
             Se un neurone non spika mai -> tempo = T.
    """
    T, B, C = spk_TBC.shape
    mask = spk_TBC > 0                          # [T,B,C]
    t_idx = torch.arange(T, device=spk_TBC.device).view(T,1,1).expand_as(spk_TBC)
    # dove non c'è spike, metto T (sentinella)
    times = torch.where(mask, t_idx, torch.full_like(t_idx, T))
    first = times.min(dim=0).values             # [B,C]; se no-spike → T
    return first


def temporal_rank_order_encode(X, x_min, x_max, T, eps =1e-8):
    """
    - X: data -> [N, F]
    - x_min, x_max: min and max per feature -> [F]
    - T:  time-steps (t in [0, T-1])

    - Returns: spike_times [N, F], integer values [0, T-1]
                big value -> small t
    """
    # Normalization in [0, 1]
    x_norm = (X - x_min) / (x_max - x_min + eps)   # [N, F]

    # mapping: 1 -> t_min, 0 -> t_max
    t = (T - 1) * (1.0 - x_norm)                  # [N, F] float
    t = np.rint(t).astype(np.int64)               # arrotonda a intero
    t = np.clip(t, 0, T - 1)

    return t


def times_to_trains(spike_times, T=256, device="cpu"):
    """
    INPUT: numpy array [B, F] with integer values between [0, T-1]
    OUPUT: torch.Tensor [T, B, F] with binary values (0/1)
    """
    t = torch.as_tensor(spike_times, device=device, dtype=torch.long)  # [B, F]
    B, F = t.shape

    S = torch.zeros(T, B, F, device=device)

    # indici batch/feature
    b_ix = torch.arange(B, device=device).unsqueeze(1).expand_as(t)   # [B, F]
    f_ix = torch.arange(F, device=device).unsqueeze(0).expand_as(t)   # [B, F]

    # metti 1 nel time-step corrispondente
    S[t, b_ix, f_ix] = 1.0

    return S  # [T, B, F]
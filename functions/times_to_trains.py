import torch

def times_to_trains(spike_times, T=256, device="cpu"):
    """
    spike_times: numpy array [B, F] con valori interi in [0, T-1]
    ritorna:    torch.Tensor [T, B, F] con spike binari (0/1)
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
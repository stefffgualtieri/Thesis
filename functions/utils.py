import torch

def times_to_trains(X_times, T=257):
    '''
    Input:
    - X_times has shape: [N_samples, N_features] with values between [0, T - 1]

    Output:
    - S has shape [T, N_sample, N_features]
    '''

    t = X_times.round().clamp_(0, T-1).long()
    B, F = t.shape

    S = torch.zeros(T,B,F)
    b_ix = torch.arange(B, device=X_times.device).unsqueeze(1).expand_as(t)
    f_ix = torch.arange(F, device=X_times.device).unsqueeze(0).expand_as(t)
    S[t, b_ix, f_ix] = 1.0
    return S  # [T, B, F]

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

def to_static_seq(x_batch, T):
    return x_batch.unsqueeze(0).expand(T, -1, -1)
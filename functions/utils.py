import torch

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
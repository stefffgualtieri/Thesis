import torch

@torch.no_grad()
def spike_rate_mean(spk_TBC: torch.Tensor) -> float:
    # frazione di spike su T*B*C
    return float((spk_TBC > 0).float().mean().item())

@torch.no_grad()
def spike_rate_per_class(spk_TBC: torch.Tensor) -> torch.Tensor:
    # [C] frazione di spike per neurone output su T*B
    return (spk_TBC > 0).float().mean(dim=(0, 1))  # [C]

@torch.no_grad()
def spike_count_per_sample(spk_TBC: torch.Tensor) -> float:
    # numero medio di spike-event per campione (proxy energia)
    T, B, C = spk_TBC.shape
    return float((spk_TBC > 0).sum().item() / max(1, B))

import torch

#-----------------------
def rank_order_encoding(
    X: torch.Tensor,
    time_min: float = 0.0,
    time_max: float = 256.0
) -> torch.Tensor:
    
    '''
    Fast version: every feature of X is normalized in [0,1]
    - X: [N, F]
    - Returns: spike_times in [time_min, time_max]
    '''

    t_min = torch.as_tensor(time_min, dtype=X.dtype, device=X.device)
    t_max = torch.as_tensor(time_max, dtype=X.dtype, device=X.device)
    return t_min + (t_max - t_min) * X



#-------------------------------
def rank_order_encoding_general(
        X: torch.Tensor,
        time_min: float = 0.0,
        time_max: float = 256.0,
        eps: float = 1e-8
) -> torch.Tensor:
    '''
    General versione: for each fatures of X compute min(n) and max(N) and compute (2)
    - X: [N_samples, N_features]
    - Returns: tensor [N_samples, N_features] of spike times
    '''

    N, _ = X.max(dim=0, keepdim=True) # [1, F]
    n, _ = X.min(dim=0, keepdim=True) # [1, F]
    m = N - n

    # Avoid division by 0
    m= m.clamp_min(eps)

    t_min = torch.as_tensor(time_min, dtype=X.dtype, device=X.device)
    t_max = torch.as_tensor(time_max, dtype=X.dtype, device=X.device)

    spike_times = ((t_max - t_min)*X)/m + ((t_min * N - t_max*n)/m)
    return spike_times
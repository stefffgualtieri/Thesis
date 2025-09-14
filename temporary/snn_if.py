import torch
import torch.nn as nn
from typing import Sequence, Optional, Tuple, List

# Norse
from norse.torch.module.lif import LIFCell
from norse.torch.functional.lif import LIFParameters, LIFState


class NorseIFNet(nn.Module):
    """
    SNN feedforward con celle LIF di Norse configurate come IF:
      - tau_mem_inv = 0   (no leak)
      - tau_syn_inv = 0   (niente filtro sinaptico)
      - nessun bias nei Linear
      - soglia per layer fissata (default = 100 per layer, come nel paper)
    Forward-only: calcola i first-spike times dell'output.
    Compatibile con metaeuristiche: gestisce solo i pesi delle Linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        v_th_per_layer: Optional[Sequence[float]] = None,  # default: 100 per layer
        dtype: torch.dtype = torch.float32,
        weight_init: str = "uniform",  # "uniform" o "xavier"
        weight_scale: float = 0.5,     # range per init uniforme
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        self.L = len(dims) - 1
        assert self.L >= 1, "Serve almeno un layer."

        # Soglia per layer: default = 100 (paper)
        if v_th_per_layer is None:
            v_th_per_layer = [100.0] * self.L
        assert len(v_th_per_layer) == self.L

        # Linear senza bias
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1], bias=False) for i in range(self.L)]
        )

        # Celle Norse configurate come IF
        self.cells: nn.ModuleList[LIFCell] = nn.ModuleList()
        for l in range(self.L):
            p = LIFParameters(
                tau_syn_inv=torch.as_tensor(0.0, dtype=dtype),  # IF: no syn dynamics
                tau_mem_inv=torch.as_tensor(0.0, dtype=dtype),  # IF: no leak
                v_th=torch.as_tensor(float(v_th_per_layer[l]), dtype=dtype),
            )
            self.cells.append(LIFCell(p))

        # Inizializzazione pesi
        for lin in self.linears:
            if weight_init == "uniform":
                nn.init.uniform_(lin.weight, a=-weight_scale, b=+weight_scale)
            elif weight_init == "xavier":
                nn.init.xavier_uniform_(lin.weight)
            else:
                raise ValueError("weight_init deve essere 'uniform' o 'xavier'.")

        self.dtype = dtype

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,                   # (B, T, D_in) sequenze già codificate
        t_max: Optional[int] = 256,        # default paper-like
        return_traces: bool = False,       # opzionale: per debug
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        assert x.dim() == 3, "Input atteso (B, T, D_in)."
        B, T, Din = x.shape
        assert Din == self.linears[0].in_features, "Dim input mismatch."

        dev = x.device
        dt = self.dtype
        x = x.to(dt)

        # Stati iniziali Norse per layer
        states: List[LIFState] = [
            LIFState(
                z=torch.zeros((B, lin.out_features), device=dev, dtype=dt),
                v=torch.zeros((B, lin.out_features), device=dev, dtype=dt),
                i=torch.zeros((B, lin.out_features), device=dev, dtype=dt),
            )
            for lin in self.linears
        ]

        # Tracce opzionali
        S_traces = [torch.zeros((B, T, lin.out_features), device=dev, dtype=dt) for lin in self.linears] if return_traces else None
        V_traces = [torch.zeros((B, T, lin.out_features), device=dev, dtype=dt) for lin in self.linears] if return_traces else None

        # First-spike times output
        no_spike_val = float(T if t_max is None else t_max)
        spike_times_out = torch.full(
            (B, self.linears[-1].out_features), no_spike_val, device=dev, dtype=dt
        )
        out_fired = torch.zeros_like(spike_times_out, dtype=torch.bool)

        for t in range(T):
            s = x[:, t, :]  # input al tempo t

            for l, (lin, cell) in enumerate(zip(self.linears, self.cells)):
                i_in = lin(s)          # proiezione lineare (no bias)
                z, states[l] = cell(i_in, states[l])  # z∈{0,1}

                s = z  # feedforward spikes

                if return_traces:
                    S_traces[l][:, t, :] = z
                    V_traces[l][:, t, :] = states[l].v

                if l == self.L - 1:
                    fired = (z > 0.0)
                    new = fired & (~out_fired)
                    if new.any():
                        spike_times_out[new] = float(t)
                        out_fired |= new

        if return_traces:
            return spike_times_out, S_traces, V_traces
        return spike_times_out

    # ---------- Utility per metaeuristiche (solo pesi) ----------
    def num_params(self) -> int:
        return sum(l.weight.numel() for l in self.linears)

    @torch.no_grad()
    def assign_from_flat(self, w_flat: torch.Tensor):
        """Assegna pesi concatenati: [W0, W1, ..., W_{L-1}] (nessun bias)."""
        assert w_flat.dim() == 1
        dev = self.linears[0].weight.device
        dt = self.linears[0].weight.dtype
        w_flat = w_flat.to(device=dev, dtype=dt)
        idx = 0
        for lin in self.linears:
            nW = lin.weight.numel()
            lin.weight.copy_(w_flat[idx:idx+nW].view_as(lin.weight))
            idx += nW
        assert idx == w_flat.numel(), "Dimensione w_flat non coerente coi pesi del modello."

    @torch.no_grad()
    def get_flat_params(self) -> torch.Tensor:
        """Ritorna i soli pesi concatenati in un vettore 1D."""
        return torch.cat([l.weight.view(-1) for l in self.linears], dim=0).clone()


# -------- Helper compatibile con la tua pipeline --------
@torch.no_grad()
def forward_to_spike_times(model: nn.Module, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Esegue solo il forward e ritorna i first-spike times dell'output.
    - X: (B, T, D_in) già codificato
    - t_max di default è 256 (coerente con il paper)
    """
    model.eval()
    X = X.to(device)
    return model(X, t_max=256)  # shape: (B, D_out)

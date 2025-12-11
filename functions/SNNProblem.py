import pygmo
import torch
import torch.nn as nn

from optimizers.utils import vector_to_weights, dim_from_layers, first_spike_times
from temporal_fitness import temporal_fitness_function

class SNNProblem:
    def __init__(
        self,
        model, 
        X, 
        y,
        layers,
        t_sim,
        tau,
        lb,
        ub,
        device="cpu"
    ):
        self.model = model
        self.X = X.to(device)
        self.y = y.to(device)
        self.lb = lb
        self.ub = ub
        self.t_sim = t_sim
        self.tau = tau
        self.device = device
        
        self.dim = dim_from_layers(layers)
    
    def get_bounds(self):
        return (self.lb, self.ub)
    
    def get_nx(self):
        return (self.dim)
    
    def get_nobj(self):
        return 1
    
    def has_gradient(self):
        return False
    
    
    def fitness(self, w_flat):
        w_flat = torch.tensor(w_flat, dtype=torch.float32, device=self.device)
        vector_to_weights(w_flat, self.layers)
        
        # Forward
        self.model.eval()
        with torch.no_grad():
            spk, _ = self.model(self.X)
            first_spikes = first_spike_times(spk)
            loss = temporal_fitness_function(
                spike_times=first_spikes,
                target_classes=self.y,
                t_sim=self.t_sim,
                tau=self.tau
            )
        return [float(loss.item())]
import torch
import torch.nn as nn
import math

from .utils import (
    vector_to_weights,
    dim_from_layers,
    get_linear_layers,
    forward_to_spike_times,
)

'''
hiking_optimization: implementation of the hiking optimization algorithm

INPUT:
    - obj_fun: objective function to evaluate the fitness of a solution proposed
    - lower_b:
    - upper_b:
    - dim: dimensionality of the solutions: for a Input+hidden+output neural network,
      we have num_features*hidden + hidden*num_class weights and "hidden + num_classes" biases
    - pop_size: number of hikers, 100 in the paper
'''

@torch.no_grad()
def hiking_optimization(
        obj_fun,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_snn: nn.Module,
        lower_b: float = 0.0,
        upper_b: float = 1.0,
        pop_size: int = 100,
        max_iter: int = 20,
        device: str ="cpu"
):         
    #Extract the layers and the dimension of the weight-vector
    layers = get_linear_layers(model_snn)
    dim = dim_from_layers(layers)

    #Pre-allocate:
    fit = torch.empty(pop_size, device=device, dtype=torch.float32)
    best_iteration = torch.empty(max_iter + 1, device=device, dtype=torch.float32)  #every iteration we have check wich is the best
    best_w = torch.zeros(dim, device=device, dtype=torch.float32)

    lb = torch.as_tensor(lower_b, device=device, dtype=torch.float32)
    ub = torch.as_tensor(upper_b, device=device, dtype=torch.float32)

    
    #Generate initial positions of the hikers: torch.rand bewteen [0,1)
    pop = lb + torch.rand(pop_size, dim, device=device) * (ub - lb)
    
    #Evaluate the initial fintess of the hikers:
    for i in range(pop_size):
        vector_to_weights(pop[i], layers)
        spike_times = forward_to_spike_times(model_snn, X_train, device)
        fit[i] = obj_fun(spike_times, y_train)
    
    #Initial Best
    best_idx = int(torch.argmin(fit))
    best_w = pop[best_idx].clone()
    best_iteration[0] = fit[best_idx].item()
    
    #Main Loop
    for i in range(1, max_iter + 1):

        #global best of current fitness
        best_idx = int(torch.argmin(fit))
        best_w = pop[best_idx].clone()

        #cycle for all the hikers
        for j in range(pop_size):

            #Initial position of the hiker
            X_ini = pop[j].clone()

            #Random angle: [0,50]
            theta_deg = torch.randint(0, 51, (1,), device=device).float()
            theta_rad = theta_deg * (math.pi / 180.0)

            #Slope and Sweep Factor
            slope = torch.tan(theta_rad)
            sweep_factor = torch.randint(1, 3, (1,), device=device).float()

            #Tobler's Hiking Function: Vel = 6 * exp(-3.5*|slope + 0.05|)
            vel = 6.0 * torch.exp(-3.5 * torch.abs(slope + 0.05))

            #actual_vel = vel + gamma * (best_w - sweep_factor * X_ini)
            gamma = torch.rand_like(X_ini)
            actual_vel = vel + gamma * (best_w - sweep_factor * X_ini)

            #calculate the new positions:
            new_pop = X_ini + actual_vel
            new_pop = torch.clamp(new_pop, lb, ub)

            #evaluate the new model
            vector_to_weights(new_pop, layers)
            spike_times = forward_to_spike_times(model_snn, X_train, device)
            f_new = obj_fun(spike_times, y_train)

            if f_new < fit[j]:
                pop[j] = new_pop
                fit[j] = f_new
        
        cur_best = fit.min()
        best_iteration[i] = cur_best

    #Return the best
    final_idx = int(torch.argmin(fit))
    best_w = pop[final_idx].clone()
    
    return best_w, best_iteration
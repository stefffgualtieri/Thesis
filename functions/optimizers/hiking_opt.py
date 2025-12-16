import torch
import torch.nn as nn
import math

from ..utils.utils import (
    vector_to_weights,
    dim_from_layers,
    get_linear_layers,
)

'''
hiking_optimization: implementation of the hiking optimization algorithm
INPUT:
    - obj_fun: objective function to evaluate the fitness of a hiker, takes as input 
               two torch.Tensor, spike_times_pred and target_class, and calculate the fitness

    - X_train: [n_samples, input_dim] tensor, the data of the training set
    - y_train: [n_samples] tensor, the correct class of each sample

    - model_snn: the neural network model

    - lower_b: lower bound
    - upper_b: upper bound

    - pop_size: number of hikers (paper: 100)
    - max_iter : maximum number of iteration of the algorithm
OUTPUT:
    - best_hiker (=solution)
    - best_fitness_value
'''

@torch.no_grad()    #don't need the gradients
def hiking_optimization(
        obj_fun,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_snn: nn.Module,
        lower_b: float = -1.0,
        upper_b: float = 1.0,
        pop_size: int = 100,
        max_iter: int = 20,
        device: str = "cpu"
):         
    model_snn.to(device)
    model_snn.eval()

    X_train.to(device)
    y_train.to(device)

    #Extract the layers and the dimension of the weight-vector
    layers = get_linear_layers(model_snn)
    #dim: num_features*hidden + hidden*num_class weights and "hidden + num_classes" biases
    dim = dim_from_layers(layers)

    #-------------------------------------------------------------------------------------
    # Pre-allocate:
    #-------------------------------------------------------------------------------------

    #for each hiker, his fitness
    fit = torch.empty(pop_size, device=device, dtype=torch.float32)

    #for each iteration, contains the fitness of the best hiker
    best_for_iteration = torch.empty(max_iter + 1, device=device, dtype=torch.float32)

    #the best hiker (=solution) found
    best_hiker = torch.zeros(dim, device=device, dtype=torch.float32)

    lb = torch.as_tensor(lower_b, device=device, dtype=torch.float32)
    ub = torch.as_tensor(upper_b, device=device, dtype=torch.float32)
    
    #-------------------------------------------------------------------------------------
    # Initialization + First Evaluation
    #-------------------------------------------------------------------------------------

    # Generate initial positions of the hikers:
    pop = lb + torch.rand(pop_size, dim, device=device) * (ub - lb)
    
    # Evaluation
    for i in range(pop_size):
        vector_to_weights(pop[i], layers)
        logits = model_snn(X_train)
        fit[i] = obj_fun(logits, y_train)
    
    # Saving initial Best
    best_idx = int(torch.argmin(fit))
    best_hiker = pop[best_idx].clone()
    best_for_iteration[0] = fit[best_idx].item()

    #-------------------------------------------------------------------------------------
    # Main Loop
    #-------------------------------------------------------------------------------------

    for i in range(1, max_iter):
        # Global Best and its index
        best_idx = int(torch.argmin(fit))
        best_hiker = pop[best_idx]

        # Cycle for all the hikers
        for j in range(pop_size):
            # Initial position of the hiker
            X_ini = pop[j]

            #Random angle: [0,50]
            theta_deg_j = torch.randint(0, 51, (1,), device=device).float()

            #Slope and Sweep Factor
            slope = torch.tan(theta_deg_j *(math.pi / 180.0))
            sweep_factor = 1.0 + 2.0 * torch.rand(1, device=device)

            # Initial Velocity with Tobler
            vel = 6.0 * torch.exp(-3.5 * torch.abs(slope + 0.05))

            # Actual Velocity
            gamma = torch.rand_like(X_ini)
            actual_vel = vel + gamma * (best_hiker - sweep_factor * X_ini)

            #calculate the new position:
            new_pos = torch.clamp(X_ini + actual_vel, lb, ub)

            # Evaluation of new position
            vector_to_weights(new_pos, layers)
            logits = model_snn(X_train)
            f_new = obj_fun(logits, y_train)

            if f_new < fit[j]:
                pop[j] = new_pos
                fit[j] = f_new

        best_for_iteration[i] = fit.min()
        if i % 10 == 0:
            print(f"Iteration {i}: {best_for_iteration[i]}")
        if i % 50 == 0:
            print(f"Nuova posizione migliore hiker: {best_hiker}")
    #-------------------------------------------------------------------------------------
    # Return the best
    #-------------------------------------------------------------------------------------
    
    final_idx = int(torch.argmin(fit))
    best_hiker = pop[final_idx].clone()

    # Copy the weights of the best hiker
    vector_to_weights(best_hiker, layers)
    
    return best_hiker, best_for_iteration
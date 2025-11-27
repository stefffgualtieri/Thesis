import torch
import torch.nn as nn
import math

from .utils import (
    vector_to_weights,
    dim_from_layers,
    get_linear_layers,
    to_static_seq
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

@torch.no_grad()
def hiking_opt_spike_static(
        obj_fun,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        model_snn: nn.Module,
        lower_b: float = -5.0,
        upper_b: float = 5.0,
        pop_size: int = 100,
        max_iter: int = 20,
        seed: int | None = None,
        device: str = "cpu",
        T: int = 256
):         
    #-------------------------------------------------------------------------------------
    # Preparation
    #-------------------------------------------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
    model_snn.eval()

    #Extract the linear layers and the dimension of the weight-vector
    layers = get_linear_layers(model_snn)
    #dim: num_features*hidden + hidden*num_class weights and "hidden + num_classes" biases
    dim = dim_from_layers(layers)

    #-------------------------------------------------------------------------------------
    #Pre-allocate:
    #-------------------------------------------------------------------------------------
    lb = torch.as_tensor(lower_b, device=device, dtype=torch.float32)
    ub = torch.as_tensor(upper_b, device=device, dtype=torch.float32)

    #for each hiker, his fitness
    fit = torch.empty(pop_size, device=device, dtype=torch.float32)

    #for each iteration, contains the fitness of the best hiker
    best_for_iteration = torch.empty(max_iter + 1, device=device, dtype=torch.float32)

    #the best hiker (=solution) found
    best_hiker = torch.zeros(dim, device=device, dtype=torch.float32)

    
    #-------------------------------------------------------------------------------------
    #Initialization
    #-------------------------------------------------------------------------------------

    #Generate random initial positions:
    pop = lb + torch.rand(pop_size, dim, device=device) * (ub - lb)

    # Initial Velocity from Tobler (scalar)
    # theta_deg = torch.randint(0, 51, (pop_size, 1), device=device, dtype=torch.int32).float()
    # theta_rad = theta_deg * (math.pi / 180.0)
    # slope = torch.tan(theta_rad)

    # v_thf = 6.0 * torch.exp(-3.5 * torch.abs(slope * 0.05))  # [pop_size,1]
    # v_thf_vec = v_thf.expand(-1, dim).clone()                      # [pop_size,dim]

    # vel = v_thf_vec.clone()
    
    #-------------------------------------------------------------------------------------
    # First iteration
    #-------------------------------------------------------------------------------------

    for i in range(pop_size):
        vector_to_weights(pop[i], layers)
        spike_times, _ = model_snn(to_static_seq(X_train, T))
        logits_tr = spike_times.sum(dim=0)
        fit[i] = obj_fun(logits_tr, y_train)
    
    #Initial Best
    best_idx = int(torch.argmin(fit))
    best_hiker = pop[best_idx].clone()
    best_for_iteration[0] = fit[best_idx].item()

    #-------------------------------------------------------------------------------------
    #Main Loop
    #-------------------------------------------------------------------------------------

    for i in range(1, max_iter):
        #global best of current fitness
        best_idx = int(torch.argmin(fit))
        best_hiker = pop[best_idx]

        # save the best
        # elite_vec = best_hiker.clone()
        # elite_fit = fit[best_idx].clone()

        #cycle for all the hikers
        for j in range(pop_size):

            # Initial position of the hiker
            X_ini = pop[j]
            # Sweep factor between [1,2]
            sweep_factor = 1.0 + 2.0 * torch.rand(1, device=device)

            # # Compute actual vel with memory:
            # gamma = torch.rand_like(X_ini)
            # vel[j] = vel[j] + gamma * (best_hiker - sweep_factor * X_ini) + v_thf_vec[j]
            # actual_vel = vel[j]

            # ---------------------------------------------------------
            # Random angle: [0°,50°]
            theta_deg_j = torch.randint(0, 51, (1,), device=device).float()
            theta_rad_j = theta_deg_j * (math.pi / 180.0)
            #Slope
            slope_j = torch.tan(theta_rad_j)
        
            # Initial velocity with Tobler
            vel = 6.0 * torch.exp(-3.5 * torch.abs(slope_j + 0.05))

            # Actual vel
            gamma = torch.rand_like(X_ini)
            actual_vel = vel + gamma * (best_hiker - sweep_factor * X_ini)

            #calculate the new position:
            new_pos = torch.clamp(X_ini + actual_vel, lb, ub)

            #evaluate the new position of the hiker
            vector_to_weights(new_pos, layers)
            spike_times, _ = model_snn(to_static_seq(X_train, T))
            logits_tr = spike_times.sum(dim=0)
            if i % 20 == 0  and j == 0:
                print(f"Spike sum: {spike_times.sum().item()}")
                print(f"Media spike per neurone di out: {logits_tr.mean(dim=0).item()}")
                # single = to_static_seq(X_train[:1], T)
                # print(f"Encoder spike sum (single sample): {single.sum(dim=0)}")
                print(f"Spikes dei primi 20 sample per neurone di output: \n{logits_tr[:20]}")
            f_new = obj_fun(logits_tr, y_train)

            if f_new <= fit[j]:
                pop[j] = new_pos
                fit[j] = f_new
        
        # worst_idx = int(torch.argmax(fit))
        # if elite_fit < fit[worst_idx]:
        #     pop[worst_idx] = elite_vec
        #     fit[worst_idx] = elite_fit
        
        # iteration best
        best_for_iteration[i] = fit.min()
        if i % 5 == 0:
            print(f"Iteration {i}: {best_for_iteration[i]}")
    
    #-------------------------------------------------------------------------------------
    # Assign the best weights to te network and return best hiker and best for each iteration
    #-------------------------------------------------------------------------------------
    
    final_idx = int(torch.argmin(fit))
    best_hiker = pop[final_idx].clone()
    vector_to_weights(best_hiker, layers)
    return best_hiker, best_for_iteration
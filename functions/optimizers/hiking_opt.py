import torch
import torch.nn as nn
import math

from utils import vector_to_weights, dim_from_layers, get_linear_layers, forward_to_spike_times
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

def hiking_optimization(
        obj_fun,
        X_train,
        y_train,
        model_snn,
        lower_b=-1,
        upper_b=1,
        dim=100,
        pop_size=100,
        max_iter=20,
        device="cpu"
):         
    #Pre-allocate:
    fit = torch.zeros(pop_size, device=device, dtype=torch.float32)
    best_iteration = torch.zeros(max_iter+1, device=device, dtype=torch.float32)  #every iteration we have check wich is the best
    best_w = torch.zeros(dim, device=device, dtype=torch.float32)

    lb = torch.as_tensor(lower_b, device=device, dtype=torch.float32)
    ub = torch.as_tensor(upper_b, device=device, dtype=torch.float32)

    
    #Generate initial positions of the hikers: torch.rand bewteen [0,1)
    delta = torch.rand(pop_size, dim, device=device)

    pop = lb + delta * (ub - lb)
    print(pop)

    #Generate the initial velocity
    velocity = torch.zeros_like(pop)


    #Extract the layers and the dimension of the model
    layers = get_linear_layers(model_snn)
    
    #Evaluate the initial fintess of the hikers:
    for i in range(pop_size):
        
        #assign the weights to the network
        vector_to_weights(pop[i], layers)
        #calculate the output from the newtork
        spike_times = forward_to_spike_times(model_snn, X_train)
        #Calculate the fitness for each individual
        fit[i] = obj_fun(spike_times, y_train)
    
    #index of the best
    best_idx = int(torch.argmin(fit))
    #best fitness value
    best_fit = fit[best_idx].item()
    best_iteration[0] = best_fit
    #Best hikers
    best_w = pop[best_idx].clone()

    for i in range(1, max_iter + 1):

        #global best of current fitness
        best_idx = int(torch.argmin(fit))
        best_w = pop[best_idx].clone()
        # best_fit=fit[best_idx].item()
        # best_iteration[i] = best_fit

        #cycle for all the hikers
        for j in range(pop_size):

            #get the initial posizione of the hiker
            X_ini = pop[j].clone()

            #get a random angle between [0,50]
            theta_deg = torch.randint(0, 51, (1,), device=device)
            theta_rad = theta_deg.float() * (math.pi / 180.0)

            #calculate the slope and the sweep factor
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
            spike_times = forward_to_spike_times(model_snn, X_train)
            f_new = obj_fun(spike_times, y_train)

            if f_new < fit[j]:
                pop[j] = new_pop
                fit[j] = f_new

        best_iteration[i+1] = torch.min(fit)
    
    return best_iteration





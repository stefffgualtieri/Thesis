import numpy as np

def hiking_optimization(obj_fun, lb, ub, dim, population_size, max_iterations):
    """
    Hiking Optimization Algorithm (HOA) - Traduzione Python dalla versione MATLAB v2 (Tobler's Hiking Function)
    
    Args:
        obj_fun (callable): funzione obiettivo che riceve un array 1D (posizione) e restituisce fitness (float)
        lb (float or array-like): limite inferiore
        ub (float or array-like): limite superiore
        dim (int): dimensione del problema (numero di variabili)
        population_size (int): numero di "hikers" (soluzioni nella popolazione)
        max_iterations (int): numero massimo di iterazioni
    
    Returns:
        tuple: (best_position, best_fitness, fitness_per_iteration)
    """
    
    # Assicuriamoci che lb e ub siano array della giusta dimensione
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
    
    # Inizializza popolazione casuale
    pop = lb + (ub - lb) * np.random.rand(population_size, dim)
    fitness = np.array([obj_fun(ind) for ind in pop])
    
    # Salva andamento fitness
    fitness_history = [np.min(fitness)]
    
    # Ciclo principale
    for iteration in range(max_iterations):
        # Migliore della popolazione
        best_idx = np.argmin(fitness)
        X_best = pop[best_idx].copy()
        
        for j in range(population_size):
            X_ini = pop[j].copy()
            
            theta = np.random.randint(0, 51)  # angolo di elevazione casuale [0,50]
            slope = np.tan(np.deg2rad(theta))  # convertiamo in radianti per il tan
            SF = np.random.randint(1, 3)  # Sweep Factor (1 o 2)
            
            # Velocità secondo Tobler's Hiking Function
            vel = 6 * np.exp(-3.5 * abs(slope + 0.05))
            new_vel = vel + np.random.rand(dim) * (X_best - SF * X_ini)
            
            # Nuova posizione
            new_pos = X_ini + new_vel
            
            # Bound check
            new_pos = np.minimum(ub, np.maximum(lb, new_pos))
            
            # Calcola fitness
            f_new = obj_fun(new_pos)
            if f_new < fitness[j]:
                pop[j] = new_pos
                fitness[j] = f_new
        
        # Salva best fitness iterazione
        fitness_history.append(np.min(fitness))
        print(f"Iterazione {iteration+1}/{max_iterations} - Miglior fitness: {fitness_history[-1]}")
    
    # Restituisci best solution
    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], fitness_history

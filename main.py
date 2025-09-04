import torch
from functions.load_dataset import load_dataset
from functions.temporal_fitness import temporal_fitness_function
from functions.optimizers.hiking_opt import hiking_optimization

#Training iris:
X_tran, X_test, y_train, y_test = load_dataset(dataset_id=53)

#Paper values
num_iteration = 20
t_max = 256
V_th = 100
num_population = 100


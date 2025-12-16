import torch
import time
import torch.nn as nn

from functions.load_dataset import load_dataset
from functions.optimizers.hiking_spike import hiking_optimization_spike
from functions.temporal_fitness import temporal_fitness_function
from functions.utils import times_to_trains
from functions.utils.utils import forward_to_spike
from models.spike_nn import SpikeNeuralNetwork
#-------------------------------------------------------------------------------------
# Load and prepare the dataset dataset
#-------------------------------------------------------------------------------------
dataset_id = 53
X_train, X_test, y_train, y_test = load_dataset(dataset_id=dataset_id)
T = 256
X_train_tr = times_to_trains(X_train, T)
#-------------------------------------------------------------------------------------
# Define the model and the parameters
#-------------------------------------------------------------------------------------
input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = int(torch.unique(y_train).numel())


beta = 0.95
tau = 20

net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=1.0
)

#-------------------------------------------------------------------------------------
# Main Training
#-------------------------------------------------------------------------------------
start_time = time.time()
best_w, best_iteration = hiking_optimization_spike(
    obj_fun=temporal_fitness_function,
    X_train=X_train_tr,
    y_train=y_train,
    model_snn=net,
    lower_b=-1,
    upper_b=1,
    pop_size=20,
    max_iter=100   
)
print(f"L'algoritmo ci ha messo: {time.time() - start_time:.3f} s")

#-------------------------------------------------------------------------------------
# Main Training
#-------------------------------------------------------------------------------------
X_test_tr = times_to_trains(X_test, T)
with torch.no_grad():
    spike_times_test = forward_to_spike(net, X_test_tr, t_sim=T)
y_pred_test = torch.argmin(spike_times_test, dim=1)
acc_test = (y_pred_test == y_test).float().mean().item()
print(f"Test accuracy (first-spike): {acc_test:.3f}")


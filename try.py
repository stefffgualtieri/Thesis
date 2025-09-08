import torch
import torch.nn as nn

from functions.load_dataset import load_dataset
from functions.temporal_fitness import temporal_fitness_function
from functions.optimizers.hiking_opt import hiking_optimization
from functions.optimizers.utils import get_linear_layers, dim_from_layers, vector_to_weights, forward_to_spike_times
from models.snn_if import NorseIFNet

torch.manual_seed(42)
device = "cpu"

# Training iris:
X_train, X_test, y_train, y_test = load_dataset(dataset_id=53, time_min=0, time_max=256, test_size=0.5, random_state=42)

# Paper values
t_max = 256
V_th = 100
pop_size = 100
max_iter = 20

# ===== Modello base (run singolo iniziale) =====
input_dim = X_train.shape[1]
num_classes = int(y_train.max().item() + 1)

model = NorseIFNet(
    input_dim=4,
    hidden_dims=[10],
    output_dim=3,
    v_th_per_layer=[1.0, 1.0]
).to(device)

layers = get_linear_layers(model)
dim = dim_from_layers(layers)
obj_fun = temporal_fitness_function


best_w, curve = hiking_optimization(
    obj_fun=obj_fun,
    X_train=X_train,
    y_train=y_train,
    model_snn=model,
    lower_b=-1,
    upper_b=1,
    dim=dim,
    pop_size=pop_size,
    max_iter=max_iter,
    device=device
)
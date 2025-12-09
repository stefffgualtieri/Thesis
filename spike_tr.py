import torch

from models.spike_nn import SpikeNeuralNetwork
from functions.load_dataset import load_dataset
from functions.optimizers.utils import times_to_trains


dataset_id = 53
time_max = 256
X_train, X_test, y_train, y_test = load_dataset(
    dataset_id=dataset_id,
    time_min=0,
    time_max=time_max,
    test_size=0.2, 
    random_state=42
) 

X_train_train = times_to_trains(X_times=X_train, T = 257)
print(X_train_train.ndim)
print(X_train_train.size())
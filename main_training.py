import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import numpy as np

from models.spike_nn import SpikeNeuralNetwork
from functions.load_dataset import load_dataset
from functions.optimizers.hiking_opt import hiking_optimization
from functions.temporal_fitness import temporal_fitness_function

#Upload the dataset:
dataset_id = 53
X_train, X_test, y_train, y_test = load_dataset(dataset_id, 0, 255, 0.5, 42)

#save the number of classes and features
num_classes = int(len(np.unique(y_train)))
num_features = X_train.shape[1]

print(f"Number of classes: {num_classes}, number of features: {num_features}")
# # ============================================
# # 2. Simulazione tempi di spike (placeholder)
# # ============================================
# # Per ora generiamo spike_times casuali: shape = [n_samples, n_classes]
# # In futuro: sostituire con output della rete SNN
# train_spike_times = np.random.randint(low=0, high=256, size=(len(X_train), n_classes))

# #calculate the temporal fitness function
# fitness_value = temporal_fitness_function(train_spike_times, y_train, n_classes)
# print(f"Errore medio (fitness) sul training set: {fitness_value}")

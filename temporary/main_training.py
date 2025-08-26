import numpy as np
from old.load_datasets import load_and_encode_dataset
from temporal_fitness import temporal_fitness_function

#Upload the dataset:
dataset_name = "iris"  # "iris", "cancer", "wine"
X_train, X_test, y_train, y_test = load_and_encode_dataset(dataset_name, 0, 255, 0.5, 42)

#save the number of classes and features
n_classes = int(len(np.unique(y_train)))
n_features = X_train.shape[1]

# ============================================
# 2. Simulazione tempi di spike (placeholder)
# ============================================
# Per ora generiamo spike_times casuali: shape = [n_samples, n_classes]
# In futuro: sostituire con output della rete SNN
train_spike_times = np.random.randint(low=0, high=256, size=(len(X_train), n_classes))

#calculate the temporal fitness function
fitness_value = temporal_fitness_function(train_spike_times, y_train, n_classes)
print(f"Errore medio (fitness) sul training set: {fitness_value}")

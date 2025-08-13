import numpy as np
from loaders.load_datasets import load_and_encode_dataset
from snn.temporal_fitness import temporal_fitness_function

# ============================================
# 1. Caricamento dataset
# ============================================
dataset_name = "iris"  # "iris", "cancer", "wine"
X_train_spikes, X_test_spikes, y_train, y_test = load_and_encode_dataset(dataset_name)

n_classes = int(len(np.unique(y_train)))
n_features = X_train_spikes.shape[1]

# ============================================
# 2. Simulazione tempi di spike (placeholder)
# ============================================
# Per ora generiamo spike_times casuali: shape = [n_samples, n_classes]
# In futuro: sostituire con output della rete SNN
train_spike_times = np.random.randint(low=0, high=256, size=(len(X_train_spikes), n_classes))

# ============================================
# 3. Calcolo fitness media sul dataset
# ============================================
fitness_value = temporal_fitness_function(train_spike_times, y_train, n_classes)
print(f"Errore medio (fitness) sul training set: {fitness_value}")

import torch
import torch.nn as nn

from functions.load_dataset import load_dataset
from models.spike_nn import SpikeNeuralNetwork
from functions.utils import times_to_trains, first_spike_times
from functions.utils.utils import times_to_trains
from functions.temporal_fitness import temporal_fitness_function

#-------------------------------------------------------------------------------------
# Pre Processing
#-------------------------------------------------------------------------------------
dataset_id = 53
T = 20
X_train, X_test, y_train, y_test = load_dataset(dataset_id=dataset_id, time_max=T)

#-------------------------------------------------------------------------------------
# Defining the network and parameters
#-------------------------------------------------------------------------------------
input_dim = X_train.shape[1]
hidden_dim = 20
output_dim = int(torch.unique(y_train).numel())

beta = 0.95
tau = 20
bias = False
num_epochs = 500
threshold = 1
lr = 0.1

model_snn = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta = beta,
    threshold=threshold,
    bias=bias
)

optimizer = torch.optim.Adam(model_snn.parameters(), lr=lr)

# Calculating spikes trains
spike_train = times_to_trains(X_train, T=T)
spike_test = times_to_trains(X_test, T=T)

#-------------------------------------------------------------------------------------
# Training phase
#-------------------------------------------------------------------------------------
model_snn.train()
for epoch in range(num_epochs):

    # Forward
    spk, _ = model_snn(spike_train)
    first_spikes = first_spike_times(spk)
    loss = temporal_fitness_function(
        spike_times=first_spikes,
        target_classes=y_train,
        t_sim=T,
        tau=tau,
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
    pred_tr = logits_tr.argmax(dim=1)
    acc_tr = (pred_tr == y_train_tensor).float().mean().item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch} | loss {loss_tr.item():.4f} acc_tr {acc_tr:.3f} |")

#-------------------------------------------------------------------------------------
# Evaluation phase
#-------------------------------------------------------------------------------------
with torch.no_grad():
    pred_train = logits_train.argmax(dim=1)
    acc_train = (pred_train == y_train).float().mean().item()

    # Test
    net.eval()
    spk_test, mem_test = net(spike_test)
    logits_test = mem_test.mean(dim=0)
    loss_test = ce(logits_test, y_test)
    pred_test = logits_test.argmax(dim=1)
    acc_test = (pred_test==y_test).float().mean().item()
    net.train()

if epoch % 5 == 0 or epoch == 1:
    print(f"Epoch {epoch:03d} | "
            f"Train Loss {loss.item():.4f} Acc {acc_train:.3f} | "
            f" Test Loss {loss_test.item():.4f} Acc {acc_test:.3f}")


# print("spk2_rec:", spk.shape, spk.dtype)
# print("mem2_rec:", mem.shape, mem.dtype)
# print("Spike sample:", spk[0, 0])     # primi spike del primo sample
# print("Membrane sample:", mem[0, 0])  # potenziale iniziale
# spike_rate = spk.mean().item()
# print(f"Spike rate medio: {spike_rate:.4f}")
# #print(mem[:, 0, :5])  # primi 5 neuroni (o meno) del primo sample
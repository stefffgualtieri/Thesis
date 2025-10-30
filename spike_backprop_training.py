import torch
import torch.nn as nn

from functions.load_dataset import load_dataset
from models.spike_nn import SpikeNeuralNetwork
from functions.utils import times_to_trains, first_spike_times
from functions.temporal_fitness import temporal_fitness_function

# Loading and encoding the dataset
dataset_id = 53
X_train, X_test, y_train, y_test = load_dataset(dataset_id=dataset_id)

# Defining the network
input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 3

T = 257
t_sim = 256
tau = 20

net = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
)

# print examples
# for i in range(10):
#     print(f"Example {i}: |{X_train[i][0]:.3f}|{X_train[i][1]:.3f}|{X_train[i][2]:.3f}|{X_train[i][3]:.3f}| |{y_train[i]}|")

# Calculating spikes trains
spike_train = times_to_trains(X_train, T=T)
spike_test = times_to_trains(X_test, T=T)

net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
ce = nn.CrossEntropyLoss()

num_epochs = 500

for epoch in range(1, num_epochs + 1):
    # Forward
    spk, mem = net(spike_train)
    logits_train = mem.mean(dim=0)
    loss = ce(logits_train, y_train)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
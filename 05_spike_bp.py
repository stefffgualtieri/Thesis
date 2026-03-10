import torch
from torch import nn
import matplotlib.pyplot as plt

from models.spike_nn import SpikeNeuralNetwork

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from functions.metrics import macro_precision_recall_f1, precision_recall_f1_binary

#-------------------------------------------------------------------------------------
# Pre Processing
#-------------------------------------------------------------------------------------
random_state = 42

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

#-------------------------------------------------------------------------------------
# Defining the networks
#-------------------------------------------------------------------------------------
beta = 0.95
T = 20
epochs = 100
bias = False
lr = 0.1
threshold = 1.0

input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = int(torch.unique(y_train_tensor).numel())

model_snn = SpikeNeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    beta=beta,
    threshold=threshold,
    bias=bias
)
ce = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model_snn.parameters(), lr=lr)

out_dir = "results/iris/snn"
#-------------------------------------------------------------------------------------
# Training Loop
#-------------------------------------------------------------------------------------

train_loss_arr = []
train_acc_arr = []
train_energy_arr = []

model_snn.train()
for epoch in range(epochs):
    # Forward
    spk_tr, _, energy_tr = model_snn(X_train_tensor, T)
    logits_tr = spk_tr.sum(dim=0)
    loss_tr = ce(logits_tr, y_train_tensor)
    train_loss_arr.append(loss_tr.item())
    train_energy_arr.append(energy_tr.detach().item())

    # Backward
    opt.zero_grad()
    loss_tr.backward()
    opt.step()
    
    
    # Forward
    with torch.no_grad():  
        pred_tr = logits_tr.argmax(dim=1)
        acc_tr = (pred_tr == y_train_tensor).float().mean().item()
        train_acc_arr.append(acc_tr)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch} | loss {loss_tr.item():.4f} acc_tr {acc_tr:.3f} |")
            

plt.figure(figsize=(8,4))
plt.plot(train_loss_arr, label="Train Loss")
plt.plot(train_acc_arr, label="Train Acc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir + "/iris_snn_bp_train_curves.png", dpi=200)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(train_energy_arr, label="Train Energy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("TRaining Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir + "/iris_snn_bp_energy.png", dpi=200)
plt.show()


#-------------------------------------------------------------------------------------
# Evaluation on test dataset
#-------------------------------------------------------------------------------------
with torch.no_grad():
    model_snn.eval()
    spk_te, _, energy_te = model_snn(X_test_tensor, T)
    logits_te = spk_te.sum(dim=0)
    test_loss = ce(logits_te, y_test_tensor).item()
    test_acc = (logits_te.argmax(dim=1) == y_test_tensor).float().mean().item()
    energy_te = energy_te.item()

    print(f"Test:\nLoss: {test_loss:.4f} | Acc: {test_acc:.4f} | ")
    
p, r, f1 = macro_precision_recall_f1(logits_te.argmax(dim=1), y_test_tensor, output_dim)

print(f"Evaluation on the test set:")
print(f"Test Loss: {test_loss}")
print(f"Test Acc: {test_acc}")
print(f"Test precision: {p}")
print(f"Test recall: {r}")
print(f"Test f1-score: {f1}")
print(f"Test Energy: {energy_te:.4f}")

with open(out_dir + "/iris_snn_bp.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation on the test set\n")
    f.write(f"Test Loss: {test_loss:.5f}\n")
    f.write(f"Test Acc: {test_acc:.5f}\n")
    f.write(f"Test Precision: {p:.5f}\n")
    f.write(f"Test Recall: {r:.5f}\n")
    f.write(f"Test f1: {f1:.5f}\n")
    f.write(f"Test Energy: {energy_te:.5f}")
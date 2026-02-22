import torch
from torch import nn
from models.classic_nn import NeuralNetwork
import matplotlib.pyplot as plt

from functions.metrics import macro_precision_recall_f1, precision_recall_f1_binary
from functions.load_adult import load_adult

#-------------------------------------------------------------------------------------
# Import dataset
#-------------------------------------------------------------------------------------
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_adult()

# Sanity Check
# print("Train:", X_train_tensor.shape, y_train_tensor.shape)
# print("Test :", X_test_tensor.shape, y_test_tensor.shape)
# print("Train positive rate:", y_train_tensor.float().mean().item())
# print("Test  positive rate:", y_test_tensor.float().mean().item())


#-------------------------------------------------------------------------------------
# Prepare the model
#-------------------------------------------------------------------------------------
input_dim = X_train_tensor.size(dim=1)
hidden_dim = 32
output_dim = int(torch.unique(y_train_tensor).numel())
bias = True

model = NeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    bias=bias
)

lr = 0.01  #learning_rate
epochs = 100 #number of epochs

# CrossEntropyLoss: wants logits values ([batch_size, num_classes]) as predictions and a y_train ([batch_size]) as target
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#-------------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------------
train_loss_arr = []
train_acc_arr = []

model.train()
for epoch in range(epochs):
    # Forward
    train_logits = model(X_train_tensor)
    current_loss = loss_fn(train_logits, y_train_tensor)
    train_loss_arr.append(current_loss.item())
    
    # Prediction
    train_pred = torch.argmax(train_logits, dim=1)
    train_correct = (y_train_tensor == train_pred).sum().item()
    train_acc = train_correct/len(y_train_tensor)
    train_acc_arr.append(train_acc)

    #zero the gradients to avoid accomulation
    optimizer.zero_grad()

    # Backward
    current_loss.backward()
    optimizer.step()
    
out_dir = "results/adult"

# Showing the graph
plt.figure(figsize=(8,4))
plt.plot(train_loss_arr, label="Train Loss")
plt.plot(train_acc_arr, label="Train Acc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir + "/classic_bp_train_curves.png", dpi=200)
plt.show()

#-------------------------------------------------------------------------------------
# Final evaluation
#-------------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    # Forward
    test_logits = model(X_test_tensor)
    test_loss = loss_fn(test_logits, y_test_tensor).item()

    # Prediction
    test_pred = torch.argmax(test_logits, dim=1)
    test_correct = (test_pred == y_test_tensor).sum().item()
    test_acc = test_correct/len(y_test_tensor)

p, r, f1 = precision_recall_f1_binary(test_pred, y_test_tensor)

print(f"Evaluation on the test set:")
print(f"Test Loss: {test_loss}")
print(f"Test Acc: {test_acc}")
print(f"Test precision: {p}")
print(f"Test recall: {r}")
print(f"Test f1-score: {f1}")

with open(out_dir + "/adult_classic_bp.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation on the test set\n")
    f.write(f"Test Loss: {test_loss:.5f}\n")
    f.write(f"Test Acc: {test_acc:.5f}\n")
    f.write(f"Test Precision: {p:.5f}\n")
    f.write(f"Test Recall: {r:.5f}\n")
    f.write(f"Test f1: {f1:.5f}\n")
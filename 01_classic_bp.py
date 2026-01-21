import torch
from torch import nn

from models.classic_nn import NeuralNetwork
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
# Prepare and scale the data, then convert into tensor
#-------------------------------------------------------------------------------------

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#-------------------------------------------------------------------------------------
# Inizialize the model
#-------------------------------------------------------------------------------------

input_dim = X_train_tensor.size(dim=1)
hidden_dim = 50
output_dim = int(torch.unique(y_train_tensor).numel())
bias = True

model = NeuralNetwork(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    bias=True
)

lr = 0.01  #learning_rate
epochs = 500    #number of epochs

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

    # Backward
    current_loss.backward()
    optimizer.step()
    #zero the gradients to avoid accomulation
    optimizer.zero_grad()

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

print(f"Evaluation on the test set:")
print(f"Test Loss: {test_loss}")
print(f"Test Acc: {test_acc}")
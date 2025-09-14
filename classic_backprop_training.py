import torch
from torch import nn

from models.classic_nn import NeuralNetwork
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Prepare and scale the data, then convert into tensor
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=220)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

#Inizialize the model
input_dim = X_train_tensor.size(dim=1)
output_dim = 2
model = NeuralNetwork(input_dim=input_dim, output_dim=output_dim)

lr = 0.001  #learning_rate
epochs = 400    #number of epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_arr = []

#Training
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    current_loss = loss_fn(y_pred, y_train_tensor)
    loss_arr.append(current_loss.item())
    current_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
plt.plot(loss_arr)
plt.show()

#Evaluation
model.eval()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long) 

    y_pred_test = model(X_test_tensor)

    test_loss = loss_fn(y_pred_test, y_test_tensor).item()

    y_pred_classes = torch.argmax(y_pred_test, dim=1)

    #accuracy
    correct = (y_pred_classes == y_test_tensor).sum().item()
    accuracy = correct/len(y_test_tensor)
print(f"Test Loss: {test_loss}")
print(f"Accuratezza: {accuracy}")
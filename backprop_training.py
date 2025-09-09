import torch 
from torch import nn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from models.classic_nn import NeuralNetwork

def fit(model):
    epochs = 400
    loss_arr = []
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
    
    for epoch in range(epochs):
        y_pred = model(X_tr_tensor)
        loss = loss_fn(y_pred, y_tr_tensor)
        loss_arr.append(loss.item())
        loss.backward()
        optim.step()
        optim.zero_grad()
    plt.plot(loss_arr)
    plt.show()

#----------------------------------------------------------------------
#Begin
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

X_tr_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

model = NeuralNetwork()
fit(model)

#evaluate the model
X_ts_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_pred = model(X_ts_tensor)
newytest = torch.argmax(y_test_pred, dim=1)

print("Accuracy:", accuracy_score(newytest.cpu(), y_test))
print("Confusion Matrix:\n", confusion_matrix(newytest.cpu(), y_test))
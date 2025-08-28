from functions.load_dataset import load_dataset
import torch
import numpy as np

#Test 1: iris
X_train, X_test, y_train, y_test = load_dataset(dataset_id=53)
print("Iris:\n")
print("Shape X:", X_train.shape)
print("Shape y:", y_train.shape)
print("Prime 5 y:", y_test[:5])
print("Classi uniche:", torch.unique(y_test))

#Test 2: wine
X_train2, X_test2, y_train2, y_test2 = load_dataset(dataset_id=109, test_size=0.41)
print("\n\nWine:\n")
print("Shape X:", X_train2.shape)
print("Shape y:", y_train2.shape)
print("Prime 5 y:", y_test2[:5])
print("Classi uniche:", torch.unique(y_test2))

#Test 3: breast_cancer
X_train3, X_test3, y_train3, y_test3 = load_dataset(dataset_id=15)
print("\n\nBreast_cancer:\n")
print("Shape X:", X_train3.shape)
print("Shape y:", y_train3.shape)
print("Prime 5 y:", y_test3[:5])
print("Classi uniche:", torch.unique(y_test3))

# #Test 4: Diabets
print("\n\nPima Indians Diabets:\n")
X_train4, X_test4, y_train4, y_test4 = load_dataset(dataset_id=0)
print("Shape X:", X_train4.shape)
print("Shape y:", y_train4.shape)
print("Prime 5 y:", y_test4[:5])
print("Classi uniche:", torch.unique(y_test4))
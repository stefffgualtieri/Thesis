import torch
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

'''
load_dataset:
    Uses a MinMaxScaler to normalize the data between [0, 1], then calls
    rank_order_encoding to encode the values with 'Rank Encoding', and finally it returns the result of 
    train_test_split (stratify=y is used to keep balance between the classies)

    Input:
        - dataset_id: iris = 53, wine = 109, diabets: 0, breast_cancer = 15, liver = 60 (ERROR)
        - time_min, time_max: the minimum and maximum times of the simulation, needed for time encoding
        - test_size: the dimensione of the test (paper valule: 50)
        
    Output:
        - 4 tensors of torch: X_train, X_test, y_train, y_test

rank_order_encoding:
    takes as input the datasets, and for each feature apply rank encoding of the paper
'''

def load_dataset(dataset_id=53, time_min = 0, time_max = 100, test_size = 0.5, random_state = 42):

    if dataset_id == 0:
        dataset = pd.read_csv("data/diabets.csv", header=None)
        X = dataset.iloc[:, :-1].to_numpy()
        y = dataset.iloc[:, -1].to_numpy()
    else:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features.to_numpy()
        y = dataset.data.targets.to_numpy().ravel()
    
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    X_spikes_times = rank_order_encoding(X_norm, time_min, time_max)

    le = LabelEncoder()
    y_num = le.fit_transform(y)

    X_train, X_test, y_train, y_test =  train_test_split(X_spikes_times, y_num, test_size=test_size, stratify=y_num, random_state=random_state)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def rank_order_encoding(X, time_min, time_max):
    spike_times = np.zeros_like(X)
    for i in range(X.shape[1]):
        x = X[:, i]
        N = np.max(x)
        n = np.min(x)
        m = N - n if N != n else 1e-8
        spike_times[:, i] = ((time_max - time_min) * x) / m  + ((time_min * N - time_max * n) / m)
    return spike_times
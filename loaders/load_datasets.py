import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''
load_and_encode_dataset: 
    Input: 
        - dataset_name: the name of the dataset you want to use, between iris, cancer and wine
        - time_min: 
        - time_max: 
        - test_size: the size of the test and train set
    
    load_and_encode_datasets: uses a MinMaxScaler to normalize the data between [0, 1], then calls
    rank_order_encoding to encode the values with 'Rank Encoding', and finalli it returns the result of 
    train_test_split (stratify=y is used to keep balance between the classies)

    rank order encoding: takes as input the datasets, and for each feature apply rank encoding of the paper
'''

def load_and_encode_dataset(dataset_name = "iris", time_min = 0, time_max = 255, test_size = 0.5, random_state = 42):
    
    if dataset_name == "iris":
        data = load_iris()
        X = data["data"]
        y = data["target"]

    elif dataset_name == "cancer":
        data = load_breast_cancer()
        X = data["data"]
        y = data["target"]
    elif dataset_name == "wine":
        data = load_wine()
        X = data["data"]
        y = data["target"]
    else:
        raise ValueError(f"Dataset '{dataset_name} not supported', Use: iris, cancer, wine")

    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
      
    X_spike_times = rank_order_encoding(X_norm, time_min, time_max)

    return train_test_split(X_spike_times, y, test_size=test_size, random_state=random_state, stratify=y)


def rank_order_encoding(X, time_min, time_max):
        spike_times = np.zeros_like(X)
        for i in range(X.shape[1]):
            x = X[:, i]
            N = np.min(x)
            n = np.max(x)
            m = n - N if N != n else 1e-8  #to avoid division by 0
            spike_times[:, i] = ((time_max - time_min) * x) / m  + ((time_min * N - time_max * n) / m)
        return spike_times
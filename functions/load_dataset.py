import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from functions.rank_order_encoding import rank_order_encoding, rank_order_encoding_general
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
'''

def load_dataset(
        dataset_id=53,
        time_min = 0,
        time_max = 256,
        test_size = 0.2,
        random_state = 42
    ):

    # ----------------------
    # Load the dataset
    # ----------------------
    if dataset_id == 0:
        dataset = pd.read_csv("data/diabets.csv", header=None)
        X = dataset.iloc[:, :-1].to_numpy()
        y = dataset.iloc[:, -1].to_numpy()
    else:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features.to_numpy()
        y = dataset.data.targets.to_numpy().ravel()
    
    # ----------------------
    # Normalization
    # ----------------------
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X).astype(np.float32)

    # ----------------------
    # Rank order encoding
    # ----------------------
    X_norm_torch = torch.from_numpy(X_norm)
    X_spikes_torch = rank_order_encoding(
        X_norm_torch, 
        time_min,
        time_max
    ).to(torch.float32)
    # back to numpy
    X_spikes = X_spikes_torch.numpy()

    # ----------------------
    # Label Encoder
    # ----------------------
    le = LabelEncoder()
    y_num = le.fit_transform(y).astype(np.int64)

    # ----------------------
    # Train Test split
    # ----------------------
    X_train, X_test, y_train, y_test =  train_test_split(
        X_spikes,
        y_num,
        test_size=test_size,
        stratify=y_num,
        random_state=random_state
    )
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
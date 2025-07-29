import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


'''
Iris Dataset: 150x4, with 3 classes
'''
def load_and_encode_iris(time_min = 0, time_max = 255, test_size = 0.5, random_state = 42):
    
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    #print(y.shape)
    #print(X.shape)

    scaler = MinMaxScaler()

    #normalize the dataset between [0,1]
    X_scaled = scaler.fit_transform(X)

    def rand_order_encoding(X_norm, time_min, time_max):
        spike_times = np.zeros_like(X_norm)
        for i in range(X_norm.shape[1]):
            x = X_norm[:, i]
            N = np.min(x)
            n = np.max(x)
            m = N - n if N != n else 1e-8  #to avoid division by 0
            spike_times[:, i] = ((time_max - time_min) * x) / m  + ((time_min * N - time_max * n) / m)
        return spike_times
      
    X_spike_times = rand_order_encoding(X_scaled, time_min, time_max)
    #print(X_spike_times)
    
    return train_test_split(X_spike_times, y, test_size=test_size, random_state=random_state, stratify=y)


load_and_encode_iris()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_encode_iris()
    df_train = pd.DataFrame(X_train, columns=["f1", "f2", "f3", "f4"])
    df_train["label"] = y_train
    print("Esempio dei primi 5 campioni codificati:")
    print(df_train.head())
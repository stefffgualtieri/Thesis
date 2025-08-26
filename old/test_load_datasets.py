import numpy as np
from old.load_datasets import load_and_encode_dataset

def test_load_and_encode_datasets():
    
    X_train, X_test, y_train, y_test = load_and_encode_dataset(
        dataset_name="iris",
        time_min=0,
        time_max=255,
        test_size=0.5,
        random_state=42
    )
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_train.shape[0]

    assert np.all(X_train >= 0) and np.all(X_train <= 255)
    assert np.all(X_test >= 0) and np.all(X_test <= 255)

    assert X_train.shape[1] == X_test.shape[1]

    assert len(np.unique(y_train)) > 1
    assert len(np.unique(y_test)) > 1
import torch
import pytest
from load_dataset import load_dataset

#Test 1: iris
def test_load_datasets_iris():
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

# (dataset_id, nome_descrittivo, kwargs aggiuntivi)
CASES = [
    (53,  "Iris",              {}),                 # test_size default nel tuo loader
    (109, "Wine",              {"test_size": 0.41}),
    (15,  "Breast Cancer",     {}),
    (0,   "Pima Indians",      {}),                 # id=0 -> CSV locale (come da tua pipeline)
]

@pytest.mark.parametrize("dataset_id,name,extra", CASES)
def test_load_dataset_shapes_and_labels(dataset_id, name, extra):
    X_train, X_test, y_train, y_test = load_dataset(dataset_id=dataset_id, **extra)

    # --- Stampe che vuoi preservare ---
    print(f"\n\n{name}:")
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Prime 5 y_test:", y_test[:5])
    print("Classi uniche in y_test:", torch.unique(y_test))

    # --- Assert minimi di sanità ---
    # Tipi
    assert isinstance(X_train, torch.Tensor)
    assert isinstance(X_test,  torch.Tensor)
    assert isinstance(y_train, torch.Tensor)
    assert isinstance(y_test,  torch.Tensor)

    # Dimensioni coerenti
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 1 and y_test.ndim == 1
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0]  == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]  # stesso n_features tra train e test

    # Classi: almeno 2 e non negative (dopo encoding)
    uniq = torch.unique(y_test)
    assert uniq.numel() >= 2
    assert torch.min(uniq) >= 0

    # Valori X in [0,1] se usi MinMaxScaler (opzionale; lascialo se sei sicuro del pre-processing)
    # assert torch.all(X_train >= 0) and torch.all(X_train <= 1)
    # assert torch.all(X_test  >= 0) and torch.all(X_test  <= 1)
    print("ciao")

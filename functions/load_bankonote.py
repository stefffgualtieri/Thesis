from ucimlrepo import fetch_ucirepo
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_banknote(
    test_size=0.2,
    random_state=42,
    standardize=True,
    device=None
):
    """
    Load Banknote Authentication dataset from ucimlrepo
    and return train/test tensors.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset
    dataset = fetch_ucirepo(id=267)

    # Features and targets
    X = dataset.data.features
    y = dataset.data.targets.squeeze()

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standardization fitted only on training set
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

    # Convert labels to numpy if needed
    y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, X_test, y_train, y_test
from ucimlrepo import fetch_ucirepo
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_digits_binary(
    digit_a=0,
    digit_b=1,
    test_size=0.2,
    random_state=42,
    standardize=True,
    device=None
):
    """
    Load Optical Recognition of Handwritten Digits from ucimlrepo,
    keep only two digits, and return train/test tensors.

    The labels are remapped as:
        digit_a -> 0
        digit_b -> 1

    Returns:
        X_train, X_test, y_train, y_test
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset
    dataset = fetch_ucirepo(id=80)

    # Features and targets
    X = dataset.data.features
    y = dataset.data.targets.squeeze()

    # Keep only the two selected digits
    mask = (y == digit_a) | (y == digit_b)
    X = X[mask]
    y = y[mask]

    # Remap labels to 0 and 1
    y = y.replace({digit_a: 0, digit_b: 1}) if hasattr(y, "replace") else y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standardization
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

    # To tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, X_test, y_train, y_test


from ucimlrepo import fetch_ucirepo
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digits_one_all(
    target_digit=0,
    test_size=0.2,
    random_state=42,
    standardize=True,
    balanced=False,
    device=None
):
    """
    Load Optical Recognition of Handwritten Digits from ucimlrepo
    and build a one-vs-rest binary classification dataset.

    Labels:
        1 -> target_digit
        0 -> all other digits

    If balanced=True, only the training set is balanced by undersampling
    the majority class.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset
    dataset = fetch_ucirepo(id=80)

    # Features and targets
    X = dataset.data.features
    y = dataset.data.targets.squeeze()

    # Binary labels: target vs rest
    y_bin = (y == target_digit).astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_bin,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bin
    )

    # Balance only the training set
    if balanced:
        train_df = pd.DataFrame(X_train).copy()
        train_df["target"] = y_train.values if hasattr(y_train, "values") else y_train

        df_pos = train_df[train_df["target"] == 1]
        df_neg = train_df[train_df["target"] == 0]

        n_min = min(len(df_pos), len(df_neg))

        df_pos = df_pos.sample(n=n_min, random_state=random_state)
        df_neg = df_neg.sample(n=n_min, random_state=random_state)

        train_df = pd.concat([df_pos, df_neg], axis=0)
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]

    # Standardization fitted only on training set
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
        X_test = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test

    # Convert labels to numpy if needed
    y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, X_test, y_train, y_test


def load_digits(
    test_size=0.2,
    random_state=42,
    standardize=True,
    device=None
):
    """
    Load Optical Recognition of Handwritten Digits dataset from ucimlrepo
    and return train/test tensors.

    Returns:
        X_train, X_test, y_train, y_test
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset
    dataset = fetch_ucirepo(id=80)

    # Features and targets as pandas objects
    X = dataset.data.features
    y = dataset.data.targets

    # Convert y to 1D numpy array
    # Sometimes targets come as a single-column DataFrame
    y = y.squeeze()

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
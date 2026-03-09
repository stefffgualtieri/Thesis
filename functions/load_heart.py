from ucimlrepo import fetch_ucirepo
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease(
    test_size=0.2,
    random_state=42,
    standardize=True,
    device=None
):
    """
    Load Heart Disease dataset from ucimlrepo and return:
        X_train, X_test, y_train, y_test

    Preprocessing:
    - replace "?" with NaN
    - drop rows with missing values
    - binarize target: 0 -> 0, 1/2/3/4 -> 1
    - one-hot encode categorical features
    - standardize only numeric features
    - stratified train/test split
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch dataset
    dataset = fetch_ucirepo(id=45)

    # Features and target
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().squeeze()

    # Merge to drop missing rows consistently
    df = X.copy()
    df["target"] = y

    # Replace possible "?" strings with NaN
    df = df.replace("?", pd.NA)

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Split features and target
    X = df.drop(columns=["target"]).copy()
    y = pd.to_numeric(df["target"])

    # Binary target: 0 -> 0, 1/2/3/4 -> 1
    y = (y > 0).astype(int)

    # Column groups
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Make sure all are numeric before encoding/scaling
    for col in categorical_cols + numeric_cols:
        X[col] = pd.to_numeric(X[col])

    # One-hot encoding for categorical features
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Standardize only the original numeric columns
    if standardize:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_test = X_test.copy()

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Convert to numpy
    X_train = X_train.to_numpy(dtype="float32")
    X_test = X_test.to_numpy(dtype="float32")
    y_train = y_train.to_numpy(dtype="int64")
    y_test = y_test.to_numpy(dtype="int64")

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train, X_test, y_train, y_test
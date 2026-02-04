from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch

def load_adult():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load Adult dataset
    adult = fetch_openml(name="adult", version=2, as_frame=True)

    X = adult.data
    y = (adult.target == ">50K").astype(int)

    # Split columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # Train / test split
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Transform
    X_train_full = preprocessor.fit_transform(X_train_full)
    X_test_full  = preprocessor.transform(X_test_full)
    # -------------------------
    # Select small subsets (1000 train, 200 test)
    # -------------------------
    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full,
        train_size=1000,
        random_state=42,
        stratify=y_train_full
    )

    X_test, _, y_test, _ = train_test_split(
        X_test_full, y_test_full,
        train_size=200,
        random_state=42,
        stratify=y_test_full
    )

    # -------------------------
    # To torch tensors
    # -------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values, dtype=torch.long, device=device)

    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test.values,  dtype=torch.long, device=device)

    return X_train, X_test, y_train, y_test

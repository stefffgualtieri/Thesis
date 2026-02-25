from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch

def load_adult_balanced():
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
    # Select small subsets
    #   - TRAIN: bilanciato (2000 classe 0 + 2000 classe 1)
    #   - TEST: stratificato (non bilanciato)
    # -------------------------

    # TRAIN bilanciato
    y_train_arr = y_train_full.to_numpy()  # y_train_full è una Series
    idx0 = (y_train_arr == 0).nonzero()[0]
    idx1 = (y_train_arr == 1).nonzero()[0]

    rng = torch.Generator().manual_seed(42)
    # se preferisci numpy: np.random.default_rng(42)

    # campiona senza rimpiazzo
    idx0_sel = torch.tensor(idx0)[torch.randperm(len(idx0), generator=rng)[:2000]].numpy()
    idx1_sel = torch.tensor(idx1)[torch.randperm(len(idx1), generator=rng)[:2000]].numpy()

    idx_sel = torch.tensor(
        torch.cat([torch.tensor(idx0_sel), torch.tensor(idx1_sel)])[torch.randperm(4000, generator=rng)]
    ).numpy()

    X_train = X_train_full[idx_sel]
    y_train = y_train_full.iloc[idx_sel]

    # TEST (lascialo stratificato/naturale)
    X_test, _, y_test, _ = train_test_split(
        X_test_full, y_test_full,
        train_size=800,
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

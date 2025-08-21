# main.py
import numpy as np
from load_datasets import load_and_encode_dataset
from hiking_optimization import hiking_optimization
from obj_fun_hiking import make_obj_fun

if __name__ == "__main__":
    np.random.seed(42)

    # 1) dati
    X_train, X_test, y_train, y_test = load_and_encode_dataset(
        dataset_name="iris", time_min=0, time_max=255, test_size=0.3, random_state=42
    )
    num_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    t_max, tau = 255, 20

    # 2) creo la obj_fun già “piena” dei dati
    obj_fun, dim, predict_times = make_obj_fun(
        X_train, y_train, num_classes,
        t_max=t_max, tau=tau, lambda_l2=1e-6, use_bias=True
    )

    # 3) bound (pesi 0..1, bias 0..t_max)
    lb = np.concatenate([
        np.full(n_features * num_classes, 0.0),        # pesi
        np.full(num_classes, 0.0)                      # bias
    ])
    ub = np.concatenate([
        np.full(n_features * num_classes, 1.0),        # pesi
        np.full(num_classes, float(t_max))             # bias
    ])

    # 4) HOA
    best_w, best_fit, hist = hiking_optimization(
        obj_fun=obj_fun,   # <- qui passa proprio la funzione ritornata
        lb=lb, ub=ub,
        dim=dim,
        population_size=30,
        max_iterations=150
    )
    print(f"\nBest fitness (train): {best_fit:.6f}")

    # 5) Valutazione sul test
    T_test = predict_times(best_w, X_test)
    y_pred = np.argmin(T_test, axis=1)
    test_acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_acc*100:.2f}%")

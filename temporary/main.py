import torch
import torch.nn as nn

from functions.load_dataset import load_dataset
from functions.temporal_fitness import temporal_fitness_function
from functions.optimizers.hiking_opt import hiking_optimization
from functions.optimizers.utils import get_linear_layers, dim_from_layers, vector_to_weights, forward_to_spike_times
from temporary.snn_if import SNN_IF

torch.manual_seed(42)
device = "cpu"

# Training iris:
X_train, X_test, y_train, y_test = load_dataset(dataset_id=53, time_min=0, time_max=256, test_size=0.5, random_state=42)

# Paper values
t_max = 256
V_th = 100
pop_size = 100
max_iter = 20

# ===== Modello base (run singolo iniziale) =====
input_dim = X_train.shape[1]
num_classes = int(y_train.max().item() + 1)
model = SNN_IF(input_dim, hidden=10, out_dim=num_classes, t_max=t_max, V_th=V_th)

layers = get_linear_layers(model)
dim = dim_from_layers(layers)

# --- OBJ: aggiungiamo una piccola "margin penalty" per separare 1° e 2° tempo ---
def obj_fun(spike_times, y_true):
    base = temporal_fitness_function(spike_times, y_true, t_sim=t_max, tau=20, device=device)
    # penalizza margine piccolo (secondo - primo) < 5
    sorted_t, _ = torch.sort(spike_times, dim=1)
    margin = sorted_t[:, 1] - sorted_t[:, 0]
    margin_penalty = torch.clamp(5.0 - margin, min=0).mean() * 0.01  # peso leggero
    return base + margin_penalty

best_w, curve = hiking_optimization(
    obj_fun=obj_fun,
    X_train=X_train,
    y_train=y_train,
    model_snn=model,
    lower_b=-1,
    upper_b=1,
    dim=dim,
    pop_size=pop_size,
    max_iter=max_iter,
    device=device
)

vector_to_weights(best_w, layers)
spk_test = forward_to_spike_times(model, X_test, device=device)

# --- Decisione "soft" per evitare pareggi su argmin ---
alpha = 15.0
scores = torch.exp(-spk_test / alpha)  # tempi piccoli -> score alto
y_pred = torch.argmax(scores, dim=1)

test_acc = (y_pred == y_test).float().mean().item()
print(f"[SINGOLO RUN] Test accuracy: {test_acc:.3f}")

# --- Diagnostica: distribuzione predizioni, tie-rate, margine, target vs non-target, no-spike ratio ---
vals, counts = torch.unique(y_pred, return_counts=True)
print("Pred class distribution:", dict(zip(vals.tolist(), counts.tolist())))

eps = 1e-6
mins = spk_test.min(dim=1, keepdim=True).values
ties = (spk_test - mins).abs() <= eps
tie_rate = (ties.sum(dim=1) > 1).float().mean().item()
print(f"Tie rate (argmin ex-aequo): {tie_rate:.3f}")

sorted_times, _ = torch.sort(spk_test, dim=1)
avg_margin = (sorted_times[:, 1] - sorted_times[:, 0]).mean().item()
print(f"Avg margin first-vs-second: {avg_margin:.4f}")

n_classes = spk_test.shape[1]
target_times = spk_test[torch.arange(len(y_test)), y_test]
non_target_mask = torch.ones_like(spk_test, dtype=torch.bool)
non_target_mask[torch.arange(len(y_test)), y_test] = False
non_target_times = spk_test[non_target_mask].view(-1, n_classes - 1)
print(f"Mean target time: {target_times.mean().item():.2f}")
print(f"Mean non-target time: {non_target_times.mean().item():.2f}")

no_spike_ratio = (spk_test == t_max).float().mean().item()
print(f"Quota di output senza spike: {no_spike_ratio:.3f}")

# ===== Preset multipli =====
presets = [
    {"name": "P1_baseline",          "V_th": 100.0, "hidden": 10, "lower_b": -0.5, "upper_b": 0.5},
    {"name": "P2_easier_threshold",  "V_th": 70.0,  "hidden": 12, "lower_b": -0.5, "upper_b": 0.5},
    {"name": "P3_wide_init",         "V_th": 100.0, "hidden": 14, "lower_b": -1.0, "upper_b": 1.0},
]

results = []

for cfg in presets:
    print(f"\n=== Running {cfg['name']} ===")
    model = SNN_IF(input_dim, hidden=cfg["hidden"], out_dim=num_classes, t_max=t_max, V_th=cfg["V_th"]).to(device)
    layers = get_linear_layers(model)
    dim = dim_from_layers(layers)

    def obj_fun(spike_times, y_true):
        base = temporal_fitness_function(spike_times, y_true, t_sim=t_max, tau=20, device=device)
        sorted_t, _ = torch.sort(spike_times, dim=1)
        margin = sorted_t[:, 1] - sorted_t[:, 0]
        margin_penalty = torch.clamp(5.0 - margin, min=0).mean() * 0.01
        return base + margin_penalty

    best_w, curve = hiking_optimization(
        obj_fun=obj_fun,
        X_train=X_train, y_train=y_train,
        model_snn=model,
        lower_b=cfg["lower_b"], upper_b=cfg["upper_b"],
        dim=dim, pop_size=pop_size, max_iter=max_iter,
        device=device
    )

    vector_to_weights(best_w, layers)
    spk_test = forward_to_spike_times(model, X_test, device=device)

    # decisione soft
    scores = torch.exp(-spk_test / 15.0)
    y_pred = torch.argmax(scores, dim=1)
    test_acc = (y_pred == y_test).float().mean().item()

    best_fit = curve[-1].item() if hasattr(curve, "__len__") else float("nan")
    print(f"{cfg['name']}: best_fitness={best_fit:.6f}, test_acc={test_acc:.3f}")

    # Diagnostica per preset (utile da tenere a vista)
    vals, counts = torch.unique(y_pred, return_counts=True)
    print("Pred class distribution:", dict(zip(vals.tolist(), counts.tolist())))
    sorted_times, _ = torch.sort(spk_test, dim=1)
    avg_margin = (sorted_times[:, 1] - sorted_times[:, 0]).mean().item()
    print(f"Avg margin first-vs-second: {avg_margin:.4f}")

    results.append({
        "name": cfg["name"],
        "V_th": cfg["V_th"],
        "hidden": cfg["hidden"],
        "lower_b": cfg["lower_b"],
        "upper_b": cfg["upper_b"],
        "best_fitness": best_fit,
        "test_acc": test_acc
    })

print("\n=== Summary ===")
for r in results:
    print(f"{r['name']}: V_th={r['V_th']}, hidden={r['hidden']}, "
          f"range=[{r['lower_b']},{r['upper_b']}], "
          f"fitness={r['best_fitness']:.6f}, acc={r['test_acc']:.3f}")

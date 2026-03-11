import numpy as np

# =========================
# OUTPUT FILE NAME
# =========================
output_dir = 'results/wine/snn/bee/'
output_file = output_dir + "bee_wine_results_summary.txt"

# =========================
# INSERT HERE YOUR 5 VALUES
# =========================
results = {
    "Accuracy":  [0.86111, 0.94444, 0.91667, 0.94444, 0.83333],
    "Precision": [0.86406, 0.95833, 0.93056, 0.95833, 0.84630],
    "Recall":    [0.85794, 0.93333, 0.90556, 0.93333, 0.83413],
    "F1-score":  [0.86099, 0.94567, 0.91789, 0.94567, 0.84017],
    "ASN":       [528.60000, 502.20000, 498.25000, 523.75000, 517.35000]
}

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Results summary\n")
    f.write("=========================\n\n")

    for metric_name, values in results.items():
        values = np.array(values, dtype=float)

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)

        f.write(f"{metric_name}\n")
        f.write(f"Values: {values.tolist()}\n")
        f.write(f"Mean: {mean_val:.5f}\n")
        f.write(f"Std:  {std_val:.5f}\n")
        f.write(f"For table: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

print(f"Results saved in '{output_file}'")
import numpy as np

# =========================
# OUTPUT FILE NAME
# =========================
output_dir = 'results/breast_cancer/snn/gaco/'
output_file = output_dir + "gaco_breast_results_summary.txt"

# =========================
# INSERT HERE YOUR 5 VALUES
# =========================
results = {
    "Accuracy":  [0.91667, 0.88889, 0.88889, 0.91667, 0.91667],
    "Precision": [0.92674, 0.90110, 0.90417, 0.92222, 0.92308],
    "Recall":    [0.91905, 0.89127, 0.88730, 0.91508, 0.92857],
    "F1-score":  [0.92288, 0.89616, 0.89565, 0.91864, 0.92582],
    "ASN":       [543.30000, 528.20000, 494.80000, 495.30000, 508.15000]
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
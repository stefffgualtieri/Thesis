import numpy as np

# =========================
# OUTPUT FILE NAME
# =========================
output_dir = 'results/heart/snn/gaco/'
output_file = output_dir + "gaco_heart_results_summary.txt"

# =========================
# INSERT HERE YOUR 5 VALUES
# =========================
results = {
    "Accuracy":  [0.81667, 0.83333, 0.80000, 0.78333, 0.85000],
    "Precision": [0.94737, 0.87500, 0.80769, 0.85714, 0.82759],
    "Recall":    [0.64286, 0.75000, 0.75000, 0.64286, 0.85714],
    "F1-score":  [0.76596, 0.80769, 0.77778, 0.73469, 0.84211],
    "ASN":       [235.85001, 254.89999, 241.64999, 228.45000, 244.89999]
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
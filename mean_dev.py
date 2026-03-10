import numpy as np

# =========================
# OUTPUT FILE NAME
# =========================
output_dir = 'results/iris/snn/gaco/'
output_file = output_dir + "gaco_iris_results_summary.txt"

# =========================
# INSERT HERE YOUR 5 VALUES
# =========================
results = {
    "Accuracy":  [0.86667, 0.86667, 0.83333, 0.96667, 0.96667],
    "Precision": [0.86742, 0.87963, 0.84982, 0.96970, 0.96970],
    "Recall":    [0.86667, 0.86667, 0.83333, 0.96667, 0.96667],
    "F1-score":  [0.86705, 0.87310, 0.84149, 0.96818, 0.96818],
    "ASN":       [321.60000, 477.70000, 297.45000, 279.65000, 279.65000]
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
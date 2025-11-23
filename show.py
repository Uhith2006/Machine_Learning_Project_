import pandas as pd

df = pd.read_csv("summary_outlier_scores.csv")

# if outlier_label exists:
if "outlier_label" in df.columns:
    outliers = df[df["outlier_label"] == -1]
else:
    # if outlier_label not present → take top 20% highest anomaly scores
    threshold = df["anomaly_score"].quantile(0.80)
    outliers = df[df["anomaly_score"] >= threshold]

print("Detected Outliers:\n")
print(outliers[["id", "sysmon_log", "generated_summary", "anomaly_score"]])

outliers.to_csv("detected_outliers.csv", index=False)
print("\nSaved outliers to detected_outliers.csv")

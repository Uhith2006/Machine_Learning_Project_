import pandas as pd

df = pd.read_csv("summary_outlier_scores.csv")

# Sort by highest anomaly score (most suspicious at top)
df_sorted = df.sort_values("anomaly_score", ascending=False)

# Show top 10 suspicious summaries
top_outliers = df_sorted.head(10)[["id", "sysmon_log", "generated_summary", "anomaly_score"]]
print(top_outliers)

# Save to file for manual review
top_outliers.to_csv("flagged_outliers_top10.csv", index=False)
print("Saved top suspicious summaries to flagged_outliers_top10.csv")

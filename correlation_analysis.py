import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("manual_review.csv")

# Convert quality to numeric scale
map_q = {"good": 2, "medium": 1, "bad": 0}
df["quality_num"] = df["manual_quality"].map(map_q)

# Correlation between anomaly score and quality
corr, p = spearmanr(df["anomaly_score"], df["quality_num"])
print(f"Spearman Correlation = {corr:.4f}")

if corr < -0.5:
    print("Strong negative correlation → Outlier detection is effective")
elif corr < -0.3:
    print("Moderate negative correlation → Useful but not perfect")
else:
    print("Weak correlation → Outlier detection didn't match quality well")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

# ======== Load generated summaries ========
df = pd.read_csv("generated_summaries.csv")

summaries = df["generated_summary"].tolist()

# ======== Step 1: Generate sentence embeddings ========
print("Creating sentence embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(summaries, convert_to_numpy=True)

# Save embeddings as a .npy file (optional)
np.save("embeddings.npy", embeddings)

# ======== Step 2: Apply Isolation Forest (Outlier Detection) ========
print("Running Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.20,   # assumes ~20% summaries may be bad
    random_state=42
)

iso.fit(embeddings)

# Higher = more anomalous
df["anomaly_score"] = -iso.decision_function(embeddings)
df["outlier_label"] = iso.predict(embeddings)   # -1 = outlier, 1 = normal

# ======== Save results to CSV ========
df.to_csv("summary_outlier_scores.csv", index=False)
print("Outlier detection complete. Results saved to summary_outlier_scores.csv")

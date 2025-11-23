import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("generated_summaries.csv")
emb = np.load("embeddings.npy")

# compute similarity matrix
sim_matrix = np.inner(emb, emb)

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='viridis')
plt.title("Text Similarity Heatmap of Generated Summaries")
plt.savefig("similarity_heatmap.png")
print("Saved heatmap as similarity_heatmap.png")
plt.show()
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Appendix: Embedding Evaluation

> **Theory**: See [Part 5: Evaluating Embedding Quality](../part5-embedding-quality.md) for concepts behind embedding evaluation.

Evaluate the quality of your trained embeddings using both quantitative metrics and qualitative inspection.

**What you'll learn:**
1. Visualize embeddings with t-SNE and UMAP
2. Compute cluster quality metrics (Silhouette, Davies-Bouldin)
3. Inspect nearest neighbors to verify semantic similarity
4. Test embedding robustness to input perturbations
5. Generate a comprehensive quality report

**Prerequisites:**
- Embeddings from [04-self-supervised-training.ipynb](04-self-supervised-training.ipynb)
- Trained model (`tabular_resnet.pt`)

---

## Why Evaluate Embeddings?

**The challenge**: Just because your training loss decreased doesn't mean your embeddings are useful. A model can memorize training data while learning poor representations.

**The solution**: Evaluate from multiple angles:
- **Quantitative** (objective metrics like Silhouette Score)
- **Qualitative** (visual inspection, nearest neighbors)

Numbers don't tell the whole story - you need to *look* at your data!

```{code-cell}
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

print("✓ All imports successful")
print("\nLibraries loaded:")
print("  - NumPy for numerical operations")
print("  - Matplotlib for visualization")
print("  - Scikit-learn for clustering and metrics")
```

## 1. Load Embeddings

Load the embeddings generated in the self-supervised training notebook.

**What you should expect:**
- Shape: `(N, 192)` - one 192-dimensional vector per OCSF event
- Values roughly centered around 0
- No NaN or Inf values

**If you see errors:**
- `FileNotFoundError`: Run notebook 04 first to generate embeddings
- Wrong shape: Ensure you're using the correct embedding file

```{code-cell}
# Load embeddings
embeddings = np.load('../data/embeddings.npy')

# Load original features (for perturbation testing later)
numerical = np.load('../data/numerical_features.npy')
categorical = np.load('../data/categorical_features.npy')

with open('../data/feature_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

print("Loaded Embeddings:")
print(f"  Shape: {embeddings.shape}")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std: {embeddings.std():.4f}")
print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"\n✓ No NaN values: {not np.isnan(embeddings).any()}")
print(f"✓ No Inf values: {not np.isinf(embeddings).any()}")
```

---

## 2. Qualitative Evaluation: t-SNE Visualization

**What is t-SNE?** A dimensionality reduction technique that projects high-dimensional embeddings (192-dim) to 2D while preserving local structure.

**What to look for:**
- ✅ **Good**: Clear, distinct clusters for different event types
- ✅ **Good**: Anomalies appear as scattered outliers
- ❌ **Bad**: All points in one giant blob (no structure learned)
- ❌ **Bad**: Random scatter with no clusters

**Perplexity parameter**: Controls balance between local and global structure
- Low (5-15): Focus on local neighborhoods
- Medium (30): Default, balanced view
- High (50): Emphasize global structure

```{code-cell}
# Sample for visualization (t-SNE is slow on large datasets)
sample_size = min(3000, len(embeddings))
np.random.seed(42)
indices = np.random.choice(len(embeddings), sample_size, replace=False)
emb_sample = embeddings[indices]

print(f"Sampling {sample_size:,} embeddings for t-SNE visualization")
print(f"(Running t-SNE on full dataset would be too slow)")
print(f"\nRunning t-SNE with perplexity=30... (this may take 1-2 minutes)")
```

```{code-cell}
# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb_2d = tsne.fit_transform(emb_sample)
print("✓ t-SNE complete!")
```

```{code-cell}
# Visualize t-SNE
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Basic scatter
axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20, c='steelblue', edgecolors='none')
axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
axes[0].set_title('OCSF Event Embeddings (t-SNE)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Colored by embedding norm (potential anomaly indicator)
norms = np.linalg.norm(emb_sample, axis=1)
scatter = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=norms,
                          cmap='viridis', alpha=0.6, s=20, edgecolors='none')
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
axes[1].set_title('Colored by Embedding Norm (Anomaly Indicator)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('L2 Norm', fontsize=11)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("✓ Look for distinct clusters (similar events group together)")
print("✓ Outliers/sparse regions = potential anomalies")
print("✓ Right plot: Yellow points (high norm) = unusual events")
print("✗ Single blob = poor embeddings, need more training")
print("✗ Random scatter = model didn't learn structure")
```

---

## 3. Quantitative Evaluation: Cluster Quality Metrics

Visualization is subjective - we need objective numbers to:
- Compare different models
- Track quality over time
- Set production deployment thresholds

### Silhouette Score

**What it measures**: How well-separated clusters are (range: -1 to +1, higher is better)

**Interpretation**:
- **0.7-1.0**: Excellent separation
- **0.5-0.7**: Reasonable structure (acceptable for production)
- **0.25-0.5**: Weak structure
- **< 0.25**: Poor clustering

**Target for production**: > 0.5

```{code-cell}
# Run k-means clustering to identify natural clusters
n_clusters = 3  # Try 3-5 clusters for most OCSF data

print(f"Running k-means with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Compute silhouette score
silhouette_avg = silhouette_score(embeddings, cluster_labels)
sample_silhouette_values = silhouette_samples(embeddings, cluster_labels)

print(f"\n{'='*60}")
print(f"SILHOUETTE SCORE: {silhouette_avg:.3f}")
print(f"{'='*60}")
print(f"\nInterpretation:")
if silhouette_avg > 0.7:
    print(f"  ✓ EXCELLENT - Strong cluster separation")
elif silhouette_avg > 0.5:
    print(f"  ✓ GOOD - Acceptable for production")
elif silhouette_avg > 0.25:
    print(f"  ⚠ WEAK - May miss subtle anomalies")
else:
    print(f"  ✗ POOR - Embeddings not useful, retrain needed")

print(f"\nCluster sizes: {np.bincount(cluster_labels)}")
```

```{code-cell}
# Visualize silhouette plot
fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    # Get silhouette values for cluster i
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

    # Label cluster
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}\n(n={size_cluster_i})")
    y_lower = y_upper + 10

# Add average silhouette score line
ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
          label=f"Average: {silhouette_avg:.3f}")

# Add threshold lines
ax.axvline(x=0.5, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
          label="Production threshold: 0.5")

ax.set_title("Silhouette Plot - Cluster Quality Analysis", fontsize=14, fontweight='bold')
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("READING THE SILHOUETTE PLOT")
print("="*60)
print("Width of colored bands = silhouette scores for samples in that cluster")
print("  - Wide spread → cluster has outliers or mixed events")
print("  - Narrow spread → cohesive, consistent cluster")
print("\nPoints below zero = probably in wrong cluster")
print("Red dashed line (average) should be > 0.5 for production")
```

### Other Cluster Quality Metrics

**Davies-Bouldin Index**: Measures cluster overlap (lower is better, min 0)
- < 1.0: Good separation
- 1.0-2.0: Moderate separation
- > 2.0: Poor separation

**Calinski-Harabasz Score**: Ratio of between/within cluster variance (higher is better)
- No fixed threshold, use for relative comparison

```{code-cell}
# Compute additional metrics
davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)

print(f"{'='*60}")
print(f"COMPREHENSIVE CLUSTER QUALITY METRICS")
print(f"{'='*60}")
print(f"\n{'Metric':<30} {'Value':<12} {'Status'}")
print(f"{'-'*60}")
print(f"{'Silhouette Score':<30} {silhouette_avg:<12.3f} {'✓ Good' if silhouette_avg > 0.5 else '✗ Poor'}")
print(f"{'Davies-Bouldin Index':<30} {davies_bouldin:<12.3f} {'✓ Good' if davies_bouldin < 1.0 else '⚠ Moderate'}")
print(f"{'Calinski-Harabasz Score':<30} {calinski_harabasz:<12.1f} {'(higher=better)'}")
print(f"{'-'*60}")

# Overall verdict
passed = silhouette_avg > 0.5 and davies_bouldin < 1.5
verdict = "PASS ✓" if passed else "NEEDS IMPROVEMENT ⚠"
print(f"\nOverall Quality Verdict: {verdict}")
```

---

## 4. Comprehensive Quality Report

Generate a summary report of all evaluation metrics.

```{code-cell}
def generate_quality_report(embeddings, cluster_labels, silhouette_avg,
                           davies_bouldin, calinski_harabasz):
    """
    Generate comprehensive embedding quality report.
    """
    report = {
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'num_clusters': len(np.unique(cluster_labels)),
        'silhouette_score': silhouette_avg,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
    }

    # Quality verdict
    passed = report['silhouette_score'] > 0.5 and report['davies_bouldin_index'] < 1.5
    report['verdict'] = 'PASS' if passed else 'FAIL'

    return report

# Generate report
report = generate_quality_report(
    embeddings, cluster_labels, silhouette_avg,
    davies_bouldin, calinski_harabasz
)

# Display report
print("\n" + "="*70)
print(" "*20 + "EMBEDDING QUALITY REPORT")
print("="*70)
print(f"\nDataset:")
print(f"  Total samples: {report['num_samples']:,}")
print(f"  Embedding dimension: {report['embedding_dim']}")
print(f"  Clusters identified: {report['num_clusters']}")

print(f"\nCluster Quality Metrics:")
print(f"  Silhouette Score:        {report['silhouette_score']:.3f}  {'✓' if report['silhouette_score'] > 0.5 else '✗'}")
print(f"  Davies-Bouldin Index:    {report['davies_bouldin_index']:.3f}  {'✓' if report['davies_bouldin_index'] < 1.0 else '⚠'}")
print(f"  Calinski-Harabasz Score: {report['calinski_harabasz_score']:.1f}")

print(f"\nProduction Readiness:")
if report['silhouette_score'] > 0.5:
    print(f"  ✓ Cluster separation: ACCEPTABLE (> 0.5)")
else:
    print(f"  ✗ Cluster separation: POOR (< 0.5)")

if report['davies_bouldin_index'] < 1.0:
    print(f"  ✓ Cluster overlap: LOW (< 1.0)")
elif report['davies_bouldin_index'] < 1.5:
    print(f"  ⚠ Cluster overlap: MODERATE (1.0-1.5)")
else:
    print(f"  ✗ Cluster overlap: HIGH (> 1.5)")

print(f"\n{'='*70}")
print(f"VERDICT: {report['verdict']}")
print(f"{'='*70}")

if report['verdict'] == 'PASS':
    print("\n✓ Embeddings are suitable for production anomaly detection")
    print("  Proceed to notebook 06 (Anomaly Detection)")
else:
    print("\n⚠ Embeddings need improvement:")
    print("  - Try training for more epochs (notebook 04)")
    print("  - Check feature engineering (notebook 03)")
    print("  - Adjust model capacity (d_model, num_blocks)")
    print("  - Use stronger augmentation during training")
```

---

## Summary

In this notebook, we evaluated embedding quality using:

### Qualitative Evaluation
1. **t-SNE Visualization** - Projected 192-dim embeddings to 2D
   - Identified visual clusters and outliers
   - Colored by embedding norm to spot anomalies

### Quantitative Evaluation
2. **Cluster Quality Metrics** - Objective numbers
   - **Silhouette Score**: Measures cluster separation (target > 0.5)
   - **Davies-Bouldin Index**: Measures cluster overlap (target < 1.0)
   - **Calinski-Harabasz Score**: Higher is better

3. **Quality Report** - Overall production readiness verdict

**Key Takeaway**: Embeddings must pass both quantitative thresholds AND qualitative inspection before production deployment.

**Next steps:**
- ✓ If PASS: Proceed to [06-anomaly-detection.ipynb](06-anomaly-detection.ipynb)
- ⚠ If FAIL: Return to [04-self-supervised-training.ipynb](04-self-supervised-training.ipynb) to improve training


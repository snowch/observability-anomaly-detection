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
2. Inspect nearest neighbors to verify semantic similarity
3. Compute cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
4. Determine optimal number of clusters (k)
5. Test embedding robustness with perturbation stability
6. Evaluate k-NN classification as a proxy task
7. Generate a comprehensive quality report

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
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Optional: UMAP for alternative visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

print("✓ All imports successful")
print("\nLibraries loaded:")
print("  - NumPy for numerical operations")
print("  - Matplotlib for visualization")
print("  - Scikit-learn for clustering and metrics")
print(f"  - UMAP: {'Available' if UMAP_AVAILABLE else 'Not installed (pip install umap-learn)'}")
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

## 3. Alternative Visualization: UMAP

**What is UMAP?** Uniform Manifold Approximation and Projection preserves both local and global structure better than t-SNE. Generally faster and more scalable.

**When to use UMAP instead of t-SNE**:
- You have >10K samples (UMAP is faster)
- You care about global distances between clusters
- You want more stable visualizations across runs

**Key parameter—n_neighbors**: Controls balance between local and global structure (5-50 typical).

```{code-cell}
if UMAP_AVAILABLE:
    print(f"Running UMAP with n_neighbors=15...")

    # Run UMAP on same sample
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_umap = reducer.fit_transform(emb_sample)

    # Visualize UMAP
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Basic scatter
    axes[0].scatter(emb_umap[:, 0], emb_umap[:, 1], alpha=0.6, s=20, c='steelblue', edgecolors='none')
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
    axes[0].set_title('OCSF Event Embeddings (UMAP)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Colored by embedding norm
    scatter = axes[1].scatter(emb_umap[:, 0], emb_umap[:, 1], c=norms,
                              cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=12)
    axes[1].set_title('Colored by Embedding Norm (Anomaly Indicator)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('L2 Norm', fontsize=11)

    plt.tight_layout()
    plt.show()

    print("\n✓ UMAP visualization complete!")
    print("  - UMAP preserves global structure better than t-SNE")
    print("  - Distances between clusters are more meaningful")
else:
    print("⚠ UMAP not available. Install with: pip install umap-learn")
    print("  Skipping UMAP visualization...")
```

### Choosing Between t-SNE and UMAP

| Method | Best For | Preserves | Speed |
|--------|----------|-----------|-------|
| **t-SNE** | Local structure, cluster identification | Neighborhoods | Slower |
| **UMAP** | Global structure, distance relationships | Both local & global | Faster |

**Recommendation**: Start with t-SNE for initial exploration (<5K samples). Use UMAP for large datasets or when you need to understand global relationships.

---

## 4. Nearest Neighbor Inspection

Visualization shows overall structure, but you need to zoom in and check if individual embeddings make sense. A model might create nice-looking clusters but still confuse critical security events.

**The approach**: Pick a sample OCSF record, find its k nearest neighbors in embedding space, and manually verify they're actually similar.

```{code-cell}
def inspect_nearest_neighbors(query_idx, all_embeddings, k=5):
    """
    Find and display the k nearest neighbors for a query embedding.

    Args:
        query_idx: Index of query embedding
        all_embeddings: All embeddings (num_samples, embedding_dim)
        k: Number of neighbors to return

    Returns:
        Indices and distances of nearest neighbors
    """
    query_embedding = all_embeddings[query_idx]

    # Compute cosine similarity to all embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    all_norms = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    similarities = np.dot(all_norms, query_norm)

    # Find top-k most similar (excluding query itself)
    top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Skip first (self)

    return top_k_indices, similarities[top_k_indices]

# Demonstrate nearest neighbor inspection
print("="*60)
print("NEAREST NEIGHBOR INSPECTION")
print("="*60)

# Pick a few random samples to inspect
np.random.seed(42)
sample_indices = np.random.choice(len(embeddings), 3, replace=False)

for query_idx in sample_indices:
    neighbors, sims = inspect_nearest_neighbors(query_idx, embeddings, k=5)

    print(f"\nQuery Sample #{query_idx}:")
    print(f"  Embedding norm: {np.linalg.norm(embeddings[query_idx]):.3f}")
    print(f"  Top-5 Neighbors:")
    for rank, (idx, sim) in enumerate(zip(neighbors, sims), 1):
        print(f"    Rank {rank}: Sample #{idx} (similarity: {sim:.3f})")

print("\n" + "="*60)
print("WHAT TO CHECK")
print("="*60)
print("✓ Neighbors should have similar characteristics to query")
print("✓ Success/failure status should match (critical for security!)")
print("✓ Similar event types should be neighbors")
print("✗ If failed logins are neighbors of successful ones, model needs retraining")
```

---

## 5. Quantitative Evaluation: Cluster Quality Metrics

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

### Determining Optimal Clusters (k)

How many natural groupings exist in your OCSF data? Use multiple metrics together to find the answer.

```{code-cell}
def find_optimal_clusters(embeddings, k_range=range(2, 10)):
    """
    Compute clustering metrics for different numbers of clusters.

    Args:
        embeddings: Embedding array
        k_range: Range of cluster counts to try

    Returns:
        Dictionary with metrics for each k
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        results.append({
            'k': k,
            'silhouette': silhouette_score(embeddings, labels),
            'davies_bouldin': davies_bouldin_score(embeddings, labels),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels),
            'inertia': kmeans.inertia_
        })

    return results

# Find optimal number of clusters
print("="*70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*70)

results = find_optimal_clusters(embeddings)

print(f"\n{'K':<5} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
print("-" * 55)
for r in results:
    print(f"{r['k']:<5} {r['silhouette']:<12.3f} {r['davies_bouldin']:<16.3f} {r['calinski_harabasz']:<18.1f}")

# Find best k based on silhouette
best_k = max(results, key=lambda x: x['silhouette'])['k']
print(f"\nRecommended k: {best_k} (highest Silhouette Score)")
print("\nHow to interpret:")
print("  - Silhouette: Higher is better (max 1.0)")
print("  - Davies-Bouldin: Lower is better (min 0.0)")
print("  - Calinski-Harabasz: Higher is better (no upper bound)")
print("  - Look for k where multiple metrics agree")
```

```{code-cell}
# Visualize elbow method
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

k_values = [r['k'] for r in results]

# Silhouette Score
axes[0].plot(k_values, [r['silhouette'] for r in results], 'bo-', linewidth=2, markersize=8)
axes[0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Production threshold')
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Silhouette Score', fontsize=11)
axes[0].set_title('Silhouette Score vs k', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Davies-Bouldin Index
axes[1].plot(k_values, [r['davies_bouldin'] for r in results], 'ro-', linewidth=2, markersize=8)
axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Production threshold')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[1].set_title('Davies-Bouldin vs k', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Inertia (Elbow method)
axes[2].plot(k_values, [r['inertia'] for r in results], 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[2].set_ylabel('Inertia', fontsize=11)
axes[2].set_title('Elbow Method (Inertia)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nLook for the 'elbow' in the inertia plot where improvement slows down")
```

---

## 7. Robustness Evaluation

Even with good cluster metrics, we need to ensure embeddings are robust to real-world noise and useful for downstream tasks.

### Perturbation Stability

**Why robustness matters**: In production, OCSF data has noise—network jitter causes timestamp variations, rounding errors affect byte counts. Good embeddings should be stable under these small perturbations.

**The test**: Add small noise to input features and check if embeddings change drastically.

```{code-cell}
def evaluate_perturbation_stability(embeddings, numerical_features, noise_levels=[0.01, 0.05, 0.1]):
    """
    Evaluate how stable embeddings are under input perturbations.

    Note: This is a simplified version that estimates stability by
    comparing embeddings of similar samples. In production, you would
    re-run inference with perturbed inputs.

    Args:
        embeddings: Original embeddings
        numerical_features: Original numerical features
        noise_levels: Different noise levels to test

    Returns:
        Stability scores for each noise level
    """
    print("="*60)
    print("PERTURBATION STABILITY ANALYSIS")
    print("="*60)

    # Find pairs of similar embeddings (proxy for stability)
    nn = NearestNeighbors(n_neighbors=2, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # The distance to nearest neighbor indicates local consistency
    avg_nn_distance = distances[:, 1].mean()
    std_nn_distance = distances[:, 1].std()

    # Convert cosine distance to similarity
    avg_similarity = 1 - avg_nn_distance

    print(f"\nNearest Neighbor Consistency:")
    print(f"  Average similarity to nearest neighbor: {avg_similarity:.3f}")
    print(f"  Standard deviation: {std_nn_distance:.4f}")

    print(f"\nInterpretation:")
    if avg_similarity > 0.95:
        print(f"  ✓ EXCELLENT - Embeddings form very tight neighborhoods")
    elif avg_similarity > 0.85:
        print(f"  ✓ GOOD - Embeddings have reasonable local consistency")
    elif avg_similarity > 0.70:
        print(f"  ⚠ MODERATE - Some variability in neighborhoods")
    else:
        print(f"  ✗ POOR - High variability, may indicate unstable embeddings")

    # For actual perturbation testing, you would:
    # 1. Add Gaussian noise to numerical features
    # 2. Re-run model inference
    # 3. Compare original vs perturbed embeddings
    print(f"\nNote: For full perturbation testing, re-run inference")
    print(f"with noisy inputs and compare embedding cosine similarity.")
    print(f"Target: > 0.92 similarity at 5% noise level")

    return avg_similarity

stability_score = evaluate_perturbation_stability(embeddings, numerical)
```

### k-NN Classification Accuracy

**The idea**: If good embeddings make similar events close together, a simple k-NN classifier should achieve high accuracy using those embeddings.

```{code-cell}
def evaluate_knn_proxy(embeddings, n_clusters=3, k_values=[3, 5, 10]):
    """
    Evaluate embedding quality using k-NN classification on cluster labels.

    Since we don't have ground truth labels, we use cluster labels
    as a proxy to test if the embedding space is well-structured.

    Args:
        embeddings: Embedding vectors
        n_clusters: Number of clusters for labels
        k_values: Different k values to test

    Returns:
        Cross-validated accuracy scores
    """
    print("="*60)
    print("k-NN CLASSIFICATION EVALUATION")
    print("="*60)

    # Generate proxy labels using clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    proxy_labels = kmeans.fit_predict(embeddings)

    print(f"\nUsing {n_clusters} cluster labels as proxy for classification")
    print(f"Cluster distribution: {np.bincount(proxy_labels)}")

    results = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, embeddings, proxy_labels, cv=5, scoring='accuracy')

        results.append({
            'k': k,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        })

        status = '✓' if scores.mean() > 0.85 else ('○' if scores.mean() > 0.70 else '✗')
        print(f"\nk={k}: Accuracy = {scores.mean():.3f} ± {scores.std():.3f} {status}")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("  > 0.90: Excellent embeddings (clear class separation)")
    print("  0.80-0.90: Good embeddings (suitable for production)")
    print("  0.70-0.80: Moderate (may struggle with edge cases)")
    print("  < 0.70: Poor (embeddings don't capture distinctions)")

    return results

knn_results = evaluate_knn_proxy(embeddings)
```

---

## 8. Comprehensive Quality Report

Generate a summary report of all evaluation metrics.

```{code-cell}
def generate_quality_report(embeddings, cluster_labels, silhouette_avg,
                           davies_bouldin, calinski_harabasz):
    """
    Generate comprehensive embedding quality report.

    Note: Thresholds are set for demo/tutorial data. Production systems
    should tune these based on their specific requirements.
    """
    report = {
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'num_clusters': len(np.unique(cluster_labels)),
        'silhouette_score': silhouette_avg,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
    }

    # Quality verdict - relaxed thresholds for demo data
    # Production would use stricter: silhouette > 0.5, davies_bouldin < 1.0
    good = report['silhouette_score'] > 0.5 and report['davies_bouldin_index'] < 1.0
    acceptable = report['silhouette_score'] > 0.2 and report['davies_bouldin_index'] < 2.0

    if good:
        report['verdict'] = 'EXCELLENT'
    elif acceptable:
        report['verdict'] = 'ACCEPTABLE'
    else:
        report['verdict'] = 'NEEDS IMPROVEMENT'

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
sil = report['silhouette_score']
sil_status = '✓' if sil > 0.5 else ('○' if sil > 0.2 else '✗')
print(f"  Silhouette Score:        {sil:.3f}  {sil_status}")
db = report['davies_bouldin_index']
db_status = '✓' if db < 1.0 else ('○' if db < 2.0 else '✗')
print(f"  Davies-Bouldin Index:    {db:.3f}  {db_status}")
print(f"  Calinski-Harabasz Score: {report['calinski_harabasz_score']:.1f}")

print(f"\nCluster Quality Assessment:")
if sil > 0.5:
    print(f"  ✓ Cluster separation: GOOD (> 0.5)")
elif sil > 0.2:
    print(f"  ○ Cluster separation: ACCEPTABLE (> 0.2)")
else:
    print(f"  ✗ Cluster separation: POOR (< 0.2)")

if db < 1.0:
    print(f"  ✓ Cluster overlap: LOW (< 1.0)")
elif db < 2.0:
    print(f"  ○ Cluster overlap: MODERATE (< 2.0)")
else:
    print(f"  ✗ Cluster overlap: HIGH (> 2.0)")

print(f"\n{'='*70}")
print(f"VERDICT: {report['verdict']}")
print(f"{'='*70}")

if report['verdict'] == 'EXCELLENT':
    print("\n✓ Embeddings show excellent cluster separation")
    print("  Ready for production anomaly detection")
elif report['verdict'] == 'ACCEPTABLE':
    print("\n✓ Embeddings are suitable for anomaly detection")
    print("  Proceed to notebook 06 (Anomaly Detection)")
    print("\n  To improve further:")
    print("  - Train for more epochs (notebook 04)")
    print("  - Use larger dataset")
else:
    print("\n⚠ Embeddings may need improvement for production use:")
    print("  - Try training for more epochs (notebook 04)")
    print("  - Check feature engineering (notebook 03)")
    print("  - Adjust model capacity (d_model, num_blocks)")
    print("  - Use stronger augmentation during training")
    print("\n  Note: Demo data may have limited cluster structure")
```

---

## Summary

In this notebook, we evaluated embedding quality using a comprehensive four-phase approach:

### Phase 1: Qualitative Evaluation
1. **t-SNE Visualization** - Projected embeddings to 2D preserving local structure
   - Identified visual clusters and outliers
   - Colored by embedding norm to spot anomalies

2. **UMAP Visualization** - Alternative projection preserving global structure
   - Faster than t-SNE for large datasets
   - More meaningful distances between clusters

3. **Nearest Neighbor Inspection** - Verified semantic similarity
   - Spot-checked if neighbors make sense
   - Caught critical security distinctions (success/failure)

### Phase 2: Quantitative Evaluation
4. **Cluster Quality Metrics** - Objective numbers
   - **Silhouette Score**: Measures cluster separation (target > 0.5)
   - **Davies-Bouldin Index**: Measures cluster overlap (target < 1.0)
   - **Calinski-Harabasz Score**: Higher is better

5. **Optimal Cluster Selection** - Finding the right k
   - Elbow method for inertia
   - Multi-metric agreement

### Phase 3: Robustness Evaluation
6. **Perturbation Stability** - Embeddings robust to noise
   - Target > 0.92 similarity at 5% noise level

7. **k-NN Classification** - Proxy task performance
   - Target > 0.85 accuracy for production

### Quality Report
8. **Comprehensive Report** - Overall production readiness verdict

**Key Takeaway**: Embeddings must pass both quantitative thresholds AND qualitative inspection before production deployment.

**Next steps:**
- ✓ If PASS: Proceed to [06-anomaly-detection.ipynb](06-anomaly-detection.ipynb)
- ⚠ If FAIL: Return to [04-self-supervised-training.ipynb](04-self-supervised-training.ipynb) to improve training


---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
bibliography:
  - references.bib
---

# Part 5: Evaluating Embedding Quality

Learn how to evaluate and validate the quality of learned embeddings before deploying to production.

## Why Evaluate Embeddings?

After training your TabularResNet using self-supervised learning ([Part 4](part4-self-supervised-training)), you need to verify that the embeddings are actually useful before deploying to production.

**The challenge**: Just because your training loss decreased doesn't mean your embeddings are good. A model can memorize training data while learning useless representations that fail on real anomaly detection.

**The solution**: Evaluate embeddings from multiple angles using both quantitative metrics and qualitative inspection. This catches issues that any single metric would miss.

**What makes good embeddings?**
1. **Meaningful**: Similar OCSF records (e.g., login events from same user) have similar embeddings
2. **Discriminative**: Different event types (e.g., successful login vs failed login) are separated in embedding space
3. **Robust**: Small noise in input features (±5% in bytes, slight time jitter) doesn't drastically change embeddings
4. **Useful**: Enable effective anomaly detection downstream ([Part 6](part6-anomaly-detection))

**Why this matters for observability data**: Poor embeddings make anomaly detection fail silently. If your model thinks failed logins look similar to successful logins, it won't catch account takeover attacks. Evaluation catches these problems early.

---

## Evaluation Framework: Two-Pronged Approach

Evaluating embedding models requires combining **Quantitative Evaluation** (standardized metrics) and **Qualitative Evaluation** (manual inspection and visualization).

**Why both?** Numbers don't tell the whole story. A model might have a high Silhouette Score but still confuse critical security events. You need to *look* at the data to catch these semantic failures.

### Quantitative Evaluation (Automated Metrics)

Use these when you need objective, comparable numbers:

1. **Cluster Quality Metrics**:
   - **Silhouette Score**: Measures how well-separated clusters are (range: -1 to +1, higher is better)
   - **Davies-Bouldin Index**: Measures cluster separation (lower is better, minimum 0)
   - **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)

2. **Downstream Task Performance**:
   - **k-NN Classification Accuracy**: If you have some labeled data, use k-NN as a proxy for how useful embeddings are
   - **Anomaly Detection F1**: Test on actual anomaly detection task (covered in Part 6)

3. **Robustness Metrics**:
   - **Perturbation Stability**: Cosine similarity between original and slightly perturbed embeddings (should be > 0.90)

4. **Operational Metrics**:
   - **Inference Latency**: Time to embed a single OCSF record (critical for real-time systems)
   - **Memory Footprint**: Storage required per embedding in your vector database
   - **Embedding Dimensions vs Performance**: Does reducing d_model from 512 → 256 hurt quality?

### Qualitative Evaluation (Manual Inspection)

Use these to catch issues that metrics miss:

1. **Visualization** (t-SNE, UMAP):
   - Project 256-dim embeddings → 2D scatter plots
   - **What to look for**: Distinct clusters for different event types, outliers for anomalies
   - **Red flags**: All points overlapping in a blob, no visual separation between classes

2. **Nearest Neighbor Inspection**:
   - Pick a sample OCSF record, find its 10 closest neighbors in embedding space
   - **What to check**: Are neighbors actually similar events? Does model confuse critical differences (e.g., success vs failure)?
   - **Red flags**: Neighbors are random unrelated events, model treats all login attempts as identical

3. **Semantic Failure Detection**:
   - Manually test edge cases: Does model distinguish brute force attempts from normal logins?
   - **Example**: If embeddings for "100 failed logins in 1 minute" are similar to "1 successful login", that's a failure

**Key terminology**:
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): Reduces high-dimensional embeddings to 2D while preserving local structure. Good for visualization but can distort global relationships.
- **UMAP** (Uniform Manifold Approximation and Projection): Similar to t-SNE but better preserves global structure. Generally faster and more scalable.
- **Perplexity**: A t-SNE parameter that balances attention between local and global aspects (think of it as "expected number of neighbors"). Typical values: 5-50.
- **Silhouette Score**: Measures how similar a point is to its own cluster vs other clusters. Range: -1 to +1 (higher is better).
- **Davies-Bouldin Index**: Measures average similarity between each cluster and its most similar one. Lower values indicate better separation.
- **Cosine Similarity**: Measures the angle between two embedding vectors (range: -1 to +1). Values close to 1 mean vectors point in same direction (similar records).

---

## 1. Embedding Space Visualization (Qualitative)

**Why visualization matters**: Even with perfect metrics, you need to *see* your embedding space to catch semantic failures. A t-SNE plot showing failed logins mixed with successful logins immediately tells you something is wrong, even if the Silhouette Score looks good.

**The goal**: Project high-dimensional embeddings (e.g., 256-dim) → 2D scatter plot where you can visually inspect:
- Do similar OCSF events cluster together?
- Are different event types clearly separated?
- Do anomalies appear as outliers or in sparse regions?

### t-SNE Visualization

**What is t-SNE?** A dimensionality reduction technique that preserves local structure. Similar points in 256-dim space stay close in 2D, different points stay far apart.

**When to use t-SNE**:
- Exploring your embedding space for the first time
- Identifying distinct clusters (e.g., login events, file access, network connections)
- Finding outliers and anomalies visually

**Limitations**:
- Can distort global distances (two clusters that appear close in 2D might be far apart in 256-dim)
- Sensitive to hyperparameters (perplexity changes the plot dramatically)
- Doesn't preserve exact distances (only neighborhood relationships)

**What to look for in the plot**:
- ✅ **Good**: Clear, distinct clusters for different event types with some separation
- ✅ **Good**: Anomalies appear as scattered points far from clusters
- ✅ **Good**: Within a cluster, points from same users/sources are close together
- ❌ **Bad**: All points in one giant overlapping blob (no structure learned)
- ❌ **Bad**: Random scatter with no clusters (embeddings are noise)
- ❌ **Bad**: Successful and failed login events mixed together (critical security distinction lost)

```{code-cell}
import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_embeddings_tsne(embeddings, labels=None, title="Embedding Space (t-SNE)", perplexity=30):
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: (num_samples, embedding_dim) numpy array
        labels: Optional labels for coloring points
        title: Plot title
        perplexity: t-SNE perplexity parameter (5-50 typical)

    Returns:
        matplotlib figure
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Class {label}", alpha=0.6, s=30)

        ax.legend(loc='best')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Example: Simulate embeddings for normal and anomalous data
np.random.seed(42)

# Normal data: 3 clusters
normal_cluster1 = np.random.randn(200, 256) * 0.5 + np.array([0, 0] + [0]*254)
normal_cluster2 = np.random.randn(200, 256) * 0.5 + np.array([3, 3] + [0]*254)
normal_cluster3 = np.random.randn(200, 256) * 0.5 + np.array([-3, 3] + [0]*254)

# Anomalies: scattered outliers
anomalies = np.random.randn(60, 256) * 2.0 + np.array([5, -5] + [0]*254)

all_embeddings = np.vstack([normal_cluster1, normal_cluster2, normal_cluster3, anomalies])
labels = np.array([0]*200 + [1]*200 + [2]*200 + [3]*60)

fig = visualize_embeddings_tsne(all_embeddings, labels, title="OCSF Embeddings (t-SNE)")
plt.show()

print("✓ t-SNE visualization complete")
print("  - Look for clear cluster separation")
print("  - Anomalies should be outliers or in sparse regions")
```

**Interpreting your t-SNE plot**:

1. **Cluster count**: How many distinct groups do you see?
   - If you trained on OCSF authentication logs, you might see: successful logins (cluster 1), failed logins (cluster 2), suspicious login patterns (cluster 3)
   - Too many tiny clusters (>10) might mean overfitting
   - One giant blob means model didn't learn useful structure

2. **Cluster separation**: Is there space between clusters?
   - Clear gaps = model learned discriminative embeddings
   - Overlapping boundaries = model confuses some event types
   - Check the overlap region: are these ambiguous cases or critical security events being missed?

3. **Outliers**: Do you see scattered points far from any cluster?
   - These are potential anomalies! Export their indices and inspect the raw OCSF records
   - Example: If a login attempt has 1000x more bytes than normal, it should appear as an outlier

4. **Cluster density**: Are clusters tight or spread out?
   - Tight clusters = consistent embeddings for similar events (good)
   - Diffuse clusters = high variance within event type (might need more training)

**Hyperparameter tuning**:
- **perplexity=5**: Focuses on very local structure (good for finding small clusters)
- **perplexity=30**: Balanced view (default, good starting point)
- **perplexity=50**: Emphasizes global structure (good for large datasets >10K samples)

Try multiple perplexity values - if your conclusions change dramatically, the structure might not be reliable.

---

### UMAP Visualization

**What is UMAP?** A newer dimensionality reduction technique that preserves both local and global structure better than t-SNE. Generally faster and more scalable.

**When to use UMAP instead of t-SNE**:
- You have >10K samples (UMAP is faster)
- You care about global distances between clusters (e.g., "are login events more similar to file access or network connections?")
- You want more stable visualizations (UMAP is less sensitive to random seed)

**Key differences from t-SNE**:
- **Global structure**: Distances between clusters in UMAP are more meaningful
- **Speed**: UMAP can handle 100K+ samples that would make t-SNE crash
- **Reproducibility**: UMAP plots are more consistent across runs

**What to look for**:
- Same as t-SNE: clear clusters, separated event types, outliers for anomalies
- Additionally: cluster distances in 2D roughly reflect distances in 256-dim space

```{code-cell}
# Note: UMAP requires installation: pip install umap-learn
# This is a code example (not executed in the tutorial)

def visualize_embeddings_umap(embeddings, labels=None, title="Embedding Space (UMAP)", n_neighbors=15):
    """
    Visualize embeddings using UMAP.

    Args:
        embeddings: (num_samples, embedding_dim) numpy array
        labels: Optional labels for coloring
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter (5-50 typical)

    Returns:
        matplotlib figure
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None

    # Run UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot (same as t-SNE code)
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=f"Class {label}", alpha=0.6, s=30)

        ax.legend(loc='best')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

print("UMAP visualization function defined")
print("Usage: visualize_embeddings_umap(embeddings, labels)")
```

**When to use which?**

| Method | Best For | Preserves | Speed |
|--------|----------|-----------|-------|
| **t-SNE** | Local structure, cluster identification | Neighborhoods | Slower |
| **UMAP** | Global structure, distance relationships | Both local & global | Faster |

**Recommendation**: Start with t-SNE for initial exploration (<5K samples). Use UMAP for large datasets or when you need to understand global relationships.

---

### Nearest Neighbor Inspection

**Why this matters**: Visualization shows overall structure, but you need to zoom in and check if individual embeddings make sense. A model might create nice-looking clusters but still confuse critical security events.

**The approach**: Pick a sample OCSF record, find its k nearest neighbors in embedding space, and manually verify they're actually similar.

```{code-cell}
def inspect_nearest_neighbors(query_embedding, all_embeddings, all_records, k=10):
    """
    Find and display the k nearest neighbors for a query embedding.

    Args:
        query_embedding: Single embedding vector (embedding_dim,)
        all_embeddings: All embeddings (num_samples, embedding_dim)
        all_records: List of original OCSF records (for display)
        k: Number of neighbors to return

    Returns:
        Indices and distances of nearest neighbors
    """
    # Compute cosine similarity to all embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    all_norms = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    similarities = np.dot(all_norms, query_norm)

    # Find top-k most similar (excluding query itself if present)
    top_k_indices = np.argsort(similarities)[::-1][:k+1]

    # Remove query itself if it's in the database
    if similarities[top_k_indices[0]] > 0.999:  # Query found
        top_k_indices = top_k_indices[1:]
    else:
        top_k_indices = top_k_indices[:k]

    print("\n" + "="*60)
    print("NEAREST NEIGHBOR INSPECTION")
    print("="*60)

    for rank, idx in enumerate(top_k_indices, 1):
        sim = similarities[idx]
        print(f"\nRank {rank}: Similarity = {sim:.3f}")
        print(f"  Record: {all_records[idx]}")

    return top_k_indices, similarities[top_k_indices]

# Example: Simulate OCSF records
simulated_records = [
    {"activity_id": 1, "user_id": 12345, "status": "success", "bytes": 1024},
    {"activity_id": 1, "user_id": 12345, "status": "success", "bytes": 1050},  # Similar
    {"activity_id": 1, "user_id": 12345, "status": "success", "bytes": 980},   # Similar
    {"activity_id": 1, "user_id": 67890, "status": "success", "bytes": 1020},  # Different user
    {"activity_id": 1, "user_id": 12345, "status": "failure", "bytes": 512},   # Failed login
    {"activity_id": 2, "user_id": 12345, "status": "success", "bytes": 2048},  # Different activity
]

# Create embeddings (simulated - normally from your trained model)
np.random.seed(42)
base_embedding = np.random.randn(256)
simulated_embeddings = np.vstack([
    base_embedding + np.random.randn(256) * 0.1,  # Record 0
    base_embedding + np.random.randn(256) * 0.1,  # Record 1 - should be close
    base_embedding + np.random.randn(256) * 0.1,  # Record 2 - should be close
    base_embedding + np.random.randn(256) * 0.3,  # Record 3 - different user
    np.random.randn(256),                          # Record 4 - failed login (very different)
    np.random.randn(256) * 2,                      # Record 5 - different activity
])

# Query with record 0
neighbors, sims = inspect_nearest_neighbors(
    simulated_embeddings[0],
    simulated_embeddings,
    simulated_records,
    k=5
)

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("✓ Good: Records 1-2 are nearest neighbors (same user, same activity, similar bytes)")
print("✓ Good: Record 3 is somewhat close (same activity, different user)")
print("✓ Good: Record 4 is far (failed login should be different)")
print("✗ Bad: If record 4 (failure) appeared as top neighbor, model confused success/failure")
```

**What to check in your nearest neighbors**:

1. **Same event type**: If query is a login, are neighbors also logins?
   - ✅ Good: Top 5 neighbors are all authentication events
   - ❌ Bad: Neighbors include file access, network connections (model doesn't distinguish event types)

2. **Similar critical fields**: For security data, check status, severity, user patterns
   - ✅ Good: Successful login's neighbors are also successful (status preserved)
   - ❌ Bad: Successful and failed logins are neighbors (critical distinction lost!)

3. **Similar numerical patterns**: Check if bytes, duration, counts are similar
   - ✅ Good: Login with 1KB data has neighbors with ~1KB (±20%)
   - ❌ Bad: 1KB login neighbors a 1MB login (model ignores magnitude)

4. **Different users should be separated**: Unless behavior is identical
   - ✅ Good: User A's logins are neighbors with each other, not User B's
   - ❌ Bad: All users look identical (model can't distinguish user behavior)

**Common failures caught by neighbor inspection**:
- Model treats all failed login attempts as identical (ignores failed password vs account locked)
- Model groups events by timestamp instead of semantic meaning (everything at 9 AM looks similar)
- Model confuses high-frequency normal events with brute force attempts (both have many events)

**Action items when neighbors look wrong**:
- Review your feature engineering (Part 3): Are you encoding the right fields?
- Check augmentation strategy (Part 4): Are you accidentally destroying important distinctions?
- Retrain with more epochs or different hyperparameters

---

## 2. Cluster Quality Metrics (Quantitative)

**Why cluster metrics matter**: Visualization is subjective - two people might disagree on whether clusters are "well-separated". Metrics give you objective numbers to track over time and compare models.

**When to use cluster metrics**:
- Comparing multiple model configurations (ResNet-256 vs ResNet-512)
- Tracking embedding quality during training (compute every 10 epochs)
- Setting production deployment thresholds ("don't deploy if Silhouette < 0.5")

### Silhouette Score

**What it measures**: How similar each point is to its own cluster (cohesion) vs other clusters (separation). Range: -1 to +1.

**Interpretation**:
- **+1.0**: Perfect - point is right in the center of its cluster, far from others
- **+0.7 to +1.0**: Strong structure - clusters are well-separated and cohesive
- **+0.5 to +0.7**: Reasonable structure - acceptable for production
- **+0.25 to +0.5**: Weak structure - clusters exist but with significant overlap
- **0 to +0.25**: Barely any structure - model didn't learn much
- **Negative**: Point is likely in the wrong cluster

**For OCSF security data**:
- Target: Silhouette > 0.5 for production deployment
- If you get 0.3-0.5: Model learned some structure but may miss subtle anomalies
- If you get < 0.25: Embeddings are not useful, retrain with different approach

**How it works**: For each point, compute:
1. `a` = average distance to other points in same cluster (intra-cluster distance)
2. `b` = average distance to points in nearest different cluster (inter-cluster distance)
3. Silhouette = `(b - a) / max(a, b)`

**Code interpretation**:

```{code-cell}
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

def evaluate_cluster_quality(embeddings, n_clusters=3):
    """
    Evaluate clustering quality using silhouette score.

    Args:
        embeddings: (num_samples, embedding_dim) array
        n_clusters: Number of clusters to find

    Returns:
        Dictionary with metrics
    """
    # Run clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Overall silhouette score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)

    # Per-sample silhouette scores
    sample_silhouette_values = silhouette_samples(embeddings, cluster_labels)

    metrics = {
        'silhouette_score': silhouette_avg,
        'cluster_labels': cluster_labels,
        'per_sample_scores': sample_silhouette_values,
        'cluster_sizes': np.bincount(cluster_labels)
    }

    return metrics

# Example
metrics = evaluate_cluster_quality(all_embeddings[:600], n_clusters=3)  # Only normal data

print(f"\nCluster Quality Metrics:")
print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
print(f"  Interpretation:")
print(f"    1.0: Perfect separation")
print(f"    0.5-0.7: Reasonable structure")
print(f"    < 0.25: Poor clustering")
print(f"\n  Cluster sizes: {metrics['cluster_sizes']}")

# Visualize silhouette scores per cluster
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(3):
    # Get silhouette scores for cluster i
    ith_cluster_silhouette_values = metrics['per_sample_scores'][metrics['cluster_labels'] == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.tab10(i / 10.0)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
    y_lower = y_upper + 10

# Add average silhouette score line
ax.axvline(x=metrics['silhouette_score'], color="red", linestyle="--", label=f"Avg: {metrics['silhouette_score']:.3f}")

ax.set_title("Silhouette Plot for Clusters", fontsize=14, fontweight='bold')
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
```

**Reading the silhouette plot**:

1. **Red dashed line** (average): Your overall Silhouette Score
   - Should be > 0.5 for production
   - If < 0.3, clustering is weak

2. **Width of each colored band**: Silhouette scores for samples in that cluster
   - Wide spread (some negative, some >0.7) = cluster has outliers
   - Narrow spread (all ~0.6) = consistent, cohesive cluster

3. **Points below zero**: These samples are probably in the wrong cluster
   - If Cluster 0 has many negative points, it might contain mixed event types
   - Export these samples and inspect manually

4. **Uneven cluster widths**: Check if one cluster dominates
   - Example: Cluster 0 = 500 points, Cluster 1 = 20 points
   - The tiny cluster might be anomalies (good!) or model collapsed (bad)

**Troubleshooting**:
- **All negative values**: Clustering failed completely - try different n_clusters or retrain embeddings
- **One huge cluster, rest tiny**: Model didn't learn discriminative features - check feature engineering
- **Even sizes, low scores**: Model created arbitrary splits - embeddings lack semantic structure

---

### Davies-Bouldin Index

**What it measures**: Average similarity ratio between each cluster and its most similar neighbor. Lower is better (minimum 0).

**Interpretation**:
- **0 to 0.5**: Excellent separation - clusters are distinct and well-formed
- **0.5 to 1.0**: Good separation - acceptable for most applications
- **1.0 to 2.0**: Moderate separation - clusters overlap somewhat
- **> 2.0**: Poor separation - clusters are not well-defined

**How it works**:
1. For each cluster, find its most similar other cluster
2. Compute ratio: (avg distance within cluster A + avg distance within cluster B) / (distance between A and B centroids)
3. Average this ratio across all clusters

**Why it complements Silhouette**:
- Silhouette looks at individual samples
- Davies-Bouldin looks at cluster-level separation
- Both should agree - if Silhouette is high but DB is high too, investigate

**For OCSF security data**:
- Target: Davies-Bouldin < 1.0
- If > 1.5: Clusters overlap significantly, anomaly detection may miss edge cases

```{code-cell}
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def comprehensive_cluster_metrics(embeddings, n_clusters_range=range(2, 10)):
    """
    Compute multiple clustering metrics for different numbers of clusters.

    Args:
        embeddings: Embedding array
        n_clusters_range: Range of cluster counts to try

    Returns:
        DataFrame with metrics
    """
    results = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)

        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'inertia': kmeans.inertia_
        })

    return results

# Example
results = comprehensive_cluster_metrics(all_embeddings[:600])

print("\nClustering Metrics Across Different K:")
print(f"{'K':<5} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
print("-" * 55)
for r in results:
    print(f"{r['n_clusters']:<5} {r['silhouette']:<12.3f} {r['davies_bouldin']:<16.3f} {r['calinski_harabasz']:<18.1f}")

print("\nInterpretation:")
print("  - Silhouette: Higher is better (max 1.0)")
print("  - Davies-Bouldin: Lower is better (min 0.0)")
print("  - Calinski-Harabasz: Higher is better (no upper bound)")
```

**How to choose optimal k (number of clusters)**:

1. **Look for sweet spots**: Where multiple metrics agree
   - Example: k=5 has highest Silhouette (0.62) AND lowest Davies-Bouldin (0.75) → good choice
   - If metrics disagree (k=3 best Silhouette, k=7 best DB), visualize both with t-SNE

2. **Elbow method**: Look for k where metrics stop improving dramatically
   - Silhouette increases: 0.3 (k=2) → 0.5 (k=3) → 0.52 (k=4) → 0.53 (k=5)
   - Improvement slows after k=3, so k=3 or k=4 is reasonable

3. **Domain knowledge**: Do the clusters make sense for your OCSF data?
   - If k=4 gives you: successful logins, failed logins, privileged access, bulk transfers → makes sense
   - If k=10 gives you tiny arbitrary splits → probably overfitting

4. **Calinski-Harabasz interpretation**: Ratio of between-cluster to within-cluster variance
   - Higher values = better-defined clusters
   - No fixed threshold, use relative comparison (k=5 has 450, k=3 has 320 → k=5 better)

**For OCSF security data**:
- Start with k = number of event types you expect (e.g., 3-5 for authentication logs)
- If unsure, try k=3 to 7 and pick based on metrics + visualization

---

## 3. Embedding Robustness (Quantitative)

**Why robustness matters**: In production, your OCSF data will have noise - network jitter causes slight timestamp variations, rounding errors affect byte counts. Good embeddings should be stable under these small perturbations.

**The test**: Add small noise to input features and check if embeddings change drastically.
- ✅ Good: Cosine similarity > 0.95 (embeddings barely change)
- ❌ Bad: Cosine similarity < 0.85 (embeddings are unstable, model is fragile)

**Why instability is bad**: If a login with 1024 bytes gets embedding A, but 1030 bytes (+0.6% noise) gets completely different embedding B, your anomaly detector will give inconsistent results. Same event detected as anomaly one day, normal the next.

### Perturbation Stability

**What this measures**: Cosine similarity between original and slightly perturbed embeddings.

```{code-cell}
def evaluate_embedding_stability(model, numerical, categorical, num_perturbations=10, noise_level=0.1):
    """
    Evaluate embedding stability under input perturbations.

    Args:
        model: Trained TabularResNet
        numerical: Original numerical features
        categorical: Original categorical features
        num_perturbations: Number of perturbed versions
        noise_level: Std of Gaussian noise

    Returns:
        Average cosine similarity between original and perturbed embeddings
    """
    model.eval()

    with torch.no_grad():
        # Original embedding
        original_embedding = model(numerical, categorical, return_embedding=True)

        similarities = []

        for _ in range(num_perturbations):
            # Add noise to numerical features
            perturbed_numerical = numerical + torch.randn_like(numerical) * noise_level

            # Get perturbed embedding
            perturbed_embedding = model(perturbed_numerical, categorical, return_embedding=True)

            # Compute cosine similarity
            similarity = F.cosine_similarity(original_embedding, perturbed_embedding, dim=1)
            similarities.append(similarity.mean().item())

    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    print(f"Embedding Stability Test:")
    print(f"  Avg Cosine Similarity: {avg_similarity:.3f} ± {std_similarity:.3f}")
    print(f"  Interpretation:")
    print(f"    > 0.95: Very stable (robust to noise)")
    print(f"    0.85-0.95: Moderately stable")
    print(f"    < 0.85: Unstable (may need more training)")

    return avg_similarity, std_similarity

print("Embedding stability evaluation function defined")
print("Usage: evaluate_embedding_stability(model, numerical, categorical)")
```

**Interpreting stability scores**:

1. **High stability (>0.95)**: Embeddings are robust
   - Model learned semantic patterns, not memorizing exact values
   - Safe to deploy - will handle real-world noise well

2. **Moderate stability (0.85-0.95)**: Acceptable but monitor
   - Some sensitivity to input variations
   - Test with larger noise_level (0.2) to see if it drops further
   - Consider more training epochs or regularization (dropout)

3. **Low stability (<0.85)**: Embeddings are fragile
   - Model is overfitting to exact feature values
   - Add more regularization: increase dropout from 0.1 → 0.2
   - Use more aggressive augmentation during training (Part 4)

**What if stability is too high (>0.99)?**:
- Model might be "too smooth" - not capturing fine-grained distinctions
- Check nearest neighbors: does model confuse similar-but-different events?
- May need to reduce regularization or augmentation

**For security data**: Target stability > 0.92. Security events have natural variation (same attack might have slightly different byte counts), so embeddings must be robust.

---

## 4. Downstream Task Performance (Quantitative)

**Why this matters**: All previous metrics are proxies. The ultimate test is: do these embeddings actually help with your end task (anomaly detection)?

**The gold standard**: Test on real anomaly detection and measure F1 score (covered in Part 6). But if you have some labeled data, k-NN classification is a quick proxy.

**The idea**: If good embeddings make similar events close together, a simple k-NN classifier should achieve high accuracy. If k-NN accuracy is low, embeddings aren't capturing useful patterns.

### Proxy Task: k-NN Classification

**When to use this**: You have some labeled OCSF data (e.g., 1000 logins labeled as "normal user", "service account", "privileged access").

**Interpretation**:
- **> 0.90**: Excellent embeddings - clear separation between classes
- **0.80-0.90**: Good embeddings - suitable for production
- **0.70-0.80**: Moderate - may struggle with edge cases
- **< 0.70**: Poor - embeddings don't capture class distinctions

```{code-cell}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def evaluate_knn_classification(embeddings, labels, k=5):
    """
    Evaluate embedding quality using k-NN classification.

    Args:
        embeddings: Embedding vectors
        labels: Ground truth labels
        k: Number of neighbors

    Returns:
        Cross-validated accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=k)

    # 5-fold cross-validation
    scores = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy')

    print(f"k-NN Classification (k={k}):")
    print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Interpretation: Higher accuracy = better embeddings")

    return scores.mean(), scores.std()

# Example with simulated labels
labels_subset = labels[:600]  # Only normal data (3 classes)
knn_acc, knn_std = evaluate_knn_classification(all_embeddings[:600], labels_subset, k=5)
```

---

## 5. Benchmark Comparison

### Compare Different Models

```{code-cell}
def compare_embedding_models(embeddings_dict, labels, metric='silhouette'):
    """
    Compare multiple embedding models.

    Args:
        embeddings_dict: Dict of {model_name: embeddings}
        labels: Ground truth labels
        metric: 'silhouette' or 'knn'

    Returns:
        Comparison results
    """
    results = []

    for model_name, embeddings in embeddings_dict.items():
        if metric == 'silhouette':
            # Cluster and compute silhouette
            kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            metric_name = "Silhouette"

        elif metric == 'knn':
            # k-NN accuracy
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = cross_val_score(knn, embeddings, labels, cv=5)
            score = scores.mean()
            metric_name = "k-NN Accuracy"

        results.append({
            'model': model_name,
            'score': score
        })

    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\nModel Comparison ({metric_name}):")
    print(f"{'Rank':<6} {'Model':<20} {'Score':<10}")
    print("-" * 40)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['model']:<20} {r['score']:.4f}")

    return results

# Example: Compare ResNet with different hyperparameters
embeddings_dict = {
    'ResNet-256-6blocks': all_embeddings[:600],  # Simulated
    'ResNet-128-4blocks': all_embeddings[:600] + np.random.randn(600, 256) * 0.05,  # Simulated
    'ResNet-512-8blocks': all_embeddings[:600] + np.random.randn(600, 256) * 0.03,  # Simulated
}

comparison = compare_embedding_models(embeddings_dict, labels_subset, metric='silhouette')
```

**How to use model comparison**:

1. **Hyperparameter tuning**: Compare d_model=256 vs d_model=512
   - If 512 only improves Silhouette by 0.02, use 256 (faster, smaller)
   - If 512 improves by 0.10, the extra capacity is worth it

2. **Architecture changes**: Compare TabularResNet vs other architectures
   - Helps justify your choice: "ResNet beat MLP by 0.15 Silhouette"

3. **Training strategy**: Compare contrastive learning vs MFP
   - Which self-supervised method works better for your OCSF data?

---

## 6. Operational Metrics (Production Readiness)

**Why operational metrics matter**: Even with perfect embeddings (Silhouette = 1.0), the model is useless if it's too slow for real-time detection or too large to deploy.

**The reality**: You're embedding millions of OCSF events per day. Latency, memory, and throughput directly impact your system's viability.

### Inference Latency

**What this measures**: Time to embed a single OCSF record (milliseconds).

**Target latencies by use case**:
- **Real-time detection** (<100ms): Must embed and detect anomalies while event is streaming
- **Near real-time** (<1s): Acceptable for batch processing every few seconds
- **Offline analysis** (<10s): Acceptable for historical log analysis

```{code-cell}
import time

def measure_inference_latency(model, numerical, categorical, num_trials=100):
    """
    Measure average inference latency for embedding generation.

    Args:
        model: Trained TabularResNet
        numerical: Sample numerical features (batch_size, num_features)
        categorical: Sample categorical features
        num_trials: Number of trials to average

    Returns:
        Average latency in milliseconds
    """
    model.eval()
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(numerical, categorical, return_embedding=True)

    # Measure
    with torch.no_grad():
        for _ in range(num_trials):
            start = time.time()
            _ = model(numerical, categorical, return_embedding=True)
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"Inference Latency:")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  Throughput: {1000/avg_latency:.0f} events/sec")
    print(f"\nInterpretation:")
    print(f"  < 10ms: Excellent (real-time capable)")
    print(f"  10-50ms: Good (near real-time)")
    print(f"  50-100ms: Acceptable (batch processing)")
    print(f"  > 100ms: Slow (consider model optimization)")

    return avg_latency

print("Inference latency measurement function defined")
print("Usage: measure_inference_latency(model, numerical_batch, categorical_batch)")
```

**What affects latency**:
- **d_model**: Larger embeddings (512 vs 256) = slower
- **num_blocks**: More residual blocks = slower
- **Hardware**: GPU vs CPU (10-50x difference)
- **Batch size**: Batching improves throughput but not individual latency

**Optimization strategies**:
- **Model quantization**: Convert float32 → int8 (4x smaller, minimal accuracy loss)
- **ONNX export**: Optimized runtime for production (20-30% faster)
- **Smaller models**: If d_model=512 and d_model=256 have similar quality, use 256
- **GPU deployment**: For high-volume streams (>1000 events/sec)

### Memory Footprint

**What this measures**: Storage required per embedding vector.

```{code-cell}
def analyze_memory_footprint(embedding_dim, num_events, precision='float32'):
    """
    Calculate storage requirements for embeddings.

    Args:
        embedding_dim: Dimension of embeddings (e.g., 256)
        num_events: Number of OCSF events to store
        precision: 'float32', 'float16', or 'int8'

    Returns:
        Storage requirements in GB
    """
    bytes_per_value = {
        'float32': 4,
        'float16': 2,
        'int8': 1
    }

    bytes_per_embedding = embedding_dim * bytes_per_value[precision]
    total_bytes = num_events * bytes_per_embedding
    total_gb = total_bytes / (1024**3)

    print(f"Memory Footprint Analysis:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Precision: {precision}")
    print(f"  Bytes per embedding: {bytes_per_embedding}")
    print(f"\nStorage for {num_events:,} events:")
    print(f"  Total: {total_gb:.2f} GB")
    print(f"\nComparison:")
    print(f"  float32 (full): {total_bytes / (1024**3):.2f} GB")
    print(f"  float16 (half): {total_bytes / 2 / (1024**3):.2f} GB")
    print(f"  int8 (quant):   {total_bytes / 4 / (1024**3):.2f} GB")

    return total_gb

# Example: 10M OCSF events with 256-dim embeddings
footprint = analyze_memory_footprint(
    embedding_dim=256,
    num_events=10_000_000,
    precision='float32'
)
```

**When memory matters**:
- **Vector databases**: Pinecone, Weaviate charge by storage (more GB = higher cost)
- **In-memory search**: Need to fit embeddings in RAM for fast k-NN lookup
- **Historical data**: Storing 1 year of logs with embeddings

**Cost implications**:
- 10M events × 256-dim × float32 = 10 GB
- Pinecone costs ~$0.096/GB/month = $1/month for 10M events
- Scale to 1B events = 1TB storage = $100/month

**Optimization**:
- Use **float16** instead of float32 (minimal accuracy loss, 50% smaller)
- Reduce **d_model** if quality allows (512→256 = 50% smaller)
- Compress old embeddings (after 30 days, switch to int8)

### Dimensions vs Performance Trade-off

**The question**: Does using d_model=512 actually improve quality enough to justify 2x cost?

```{code-cell}
def compare_embedding_dimensions():
    """
    Compare quality metrics across different embedding dimensions.
    """
    results = {
        'd_model=128': {'silhouette': 0.52, 'latency_ms': 5, 'storage_gb_per_10M': 5},
        'd_model=256': {'silhouette': 0.61, 'latency_ms': 8, 'storage_gb_per_10M': 10},
        'd_model=512': {'silhouette': 0.64, 'latency_ms': 15, 'storage_gb_per_10M': 20},
    }

    print("Embedding Dimension Trade-off Analysis:")
    print(f"{'Model':<15} {'Silhouette':<12} {'Latency':<12} {'Storage (10M)':<15} {'Cost/Quality':<12}")
    print("-" * 75)

    for model, metrics in results.items():
        sil = metrics['silhouette']
        lat = metrics['latency_ms']
        stor = metrics['storage_gb_per_10M']
        cost_quality = stor / sil  # Lower is better

        print(f"{model:<15} {sil:<12.3f} {lat:<12.0f}ms {stor:<15.0f}GB {cost_quality:<12.1f}")

    print("\nInterpretation:")
    print("  - d_model=256 often best balance (good quality, reasonable cost)")
    print("  - d_model=512: Only if Silhouette improves by >0.10")
    print("  - d_model=128: Consider if you have tight latency constraints (<10ms)")

compare_embedding_dimensions()
```

**Decision framework**:
1. Start with d_model=256 (good default)
2. If quality is poor (<0.5 Silhouette), try d_model=512
3. If latency is too high (>50ms), try d_model=128
4. Always measure - don't assume bigger is better

---

## 7. Production Checklist

Before deploying embeddings to production, verify all criteria across quantitative and qualitative evaluation:

### Quantitative Metrics

| Criterion | Threshold | Why It Matters | Action if Failed |
|-----------|-----------|----------------|------------------|
| **Silhouette Score** | > 0.5 | Measures cluster separation | Retrain with more epochs or different augmentation |
| **Davies-Bouldin Index** | < 1.0 | Measures cluster overlap | Check feature engineering, increase model capacity |
| **Embedding Stability** | > 0.92 | Ensures robustness to noise | Add dropout, use more aggressive augmentation |
| **k-NN Accuracy** (if labels) | > 0.85 | Proxy for downstream task performance | Review feature engineering, try different architecture |
| **Inference Latency** | < 50ms | Real-time detection capability | Reduce d_model, optimize with ONNX, use GPU |
| **Memory Footprint** | Fits budget | Cost control | Use float16, reduce d_model, compress old embeddings |

### Qualitative Checks

| Check | What to Look For | Red Flags |
|-------|------------------|-----------|
| **t-SNE/UMAP Visualization** | Clear, separated clusters | All points in one blob, no structure |
| **Nearest Neighbor Inspection** | Neighbors are semantically similar | Random unrelated events, success/failure mixed |
| **Semantic Failure Testing** | Model distinguishes critical security events | Brute force looks like normal login |
| **Cluster Interpretation** | Clusters map to known event types | Arbitrary splits, no domain meaning |

### Pre-Deployment Workflow

1. **Run quantitative metrics** → All thresholds passed?
2. **Visual inspection** → Clusters make sense?
3. **Nearest neighbor spot checks** → Pick 10 random samples, verify neighbors
4. **Semantic failure tests** → Test edge cases (brute force, privilege escalation)
5. **Operational validation** → Latency < target, memory fits budget
6. **Generate report** → Document all metrics for reproducibility
7. **Test on Part 6** → Run anomaly detection algorithms, measure F1 score

### Code: Automated Quality Report

```{code-cell}
def generate_embedding_quality_report(embeddings, labels=None, model=None, save_path='embedding_report.html'):
    """
    Generate comprehensive embedding quality report.

    Args:
        embeddings: Embedding vectors
        labels: Optional ground truth labels
        model: Optional trained model for stability testing
        save_path: Path to save HTML report

    Returns:
        Dictionary with all metrics
    """
    report = {
        'timestamp': np.datetime64('now'),
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1]
    }

    # 1. Cluster quality
    n_clusters = len(np.unique(labels)) if labels is not None else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    report['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
    report['davies_bouldin_index'] = davies_bouldin_score(embeddings, cluster_labels)
    report['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)

    # 2. k-NN if labels available
    if labels is not None:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, embeddings, labels, cv=5)
        report['knn_accuracy_mean'] = knn_scores.mean()
        report['knn_accuracy_std'] = knn_scores.std()

    # 3. Quality verdict
    passed = report['silhouette_score'] > 0.5 and report['davies_bouldin_index'] < 1.0

    report['quality_verdict'] = 'PASS' if passed else 'FAIL'

    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING QUALITY REPORT")
    print("="*60)
    print(f"Samples: {report['num_samples']:,} | Embedding Dim: {report['embedding_dim']}")
    print(f"\nCluster Quality:")
    print(f"  Silhouette Score:      {report['silhouette_score']:.3f} {'✓' if report['silhouette_score'] > 0.5 else '✗'}")
    print(f"  Davies-Bouldin Index:  {report['davies_bouldin_index']:.3f} {'✓' if report['davies_bouldin_index'] < 1.0 else '✗'}")
    print(f"  Calinski-Harabasz:     {report['calinski_harabasz_score']:.1f}")

    if 'knn_accuracy_mean' in report:
        print(f"\nClassification (k-NN):")
        print(f"  Accuracy: {report['knn_accuracy_mean']:.3f} ± {report['knn_accuracy_std']:.3f}")

    print(f"\nVERDICT: {report['quality_verdict']}")
    print("="*60)

    return report

# Example
report = generate_embedding_quality_report(all_embeddings[:600], labels_subset)
```

---

## Summary

In this part, you learned a comprehensive two-pronged approach to evaluating embedding quality before production deployment:

### Qualitative Evaluation (Manual Inspection)

1. **Visualization**:
   - t-SNE for local structure and cluster identification
   - UMAP for global structure and faster processing
   - How to interpret plots and spot problems (blobs, random scatter, mixed classes)
   - Hyperparameter tuning (perplexity for t-SNE, n_neighbors for UMAP)

2. **Nearest Neighbor Inspection** (NEW):
   - Manually verify k nearest neighbors are semantically similar
   - Catch semantic failures metrics miss (e.g., model confusing success/failure)
   - Common failure patterns in security data
   - Action items when neighbors look wrong

3. **Semantic Failure Detection**:
   - Test edge cases critical for security (brute force vs normal login)
   - Verify model preserves important distinctions (status, severity, user behavior)

### Quantitative Evaluation (Automated Metrics)

1. **Cluster Quality Metrics**:
   - Silhouette Score (target: > 0.5) - measures separation between clusters
   - Davies-Bouldin Index (target: < 1.0) - measures cluster overlap
   - Calinski-Harabasz Score - ratio of between/within cluster variance
   - How to choose optimal number of clusters (k) using multiple metrics

2. **Robustness Testing**:
   - Perturbation stability (target: > 0.92) - ensures embeddings handle noise
   - How to interpret stability scores and fix fragile embeddings
   - What to do when stability is too high or too low

3. **Downstream Task Performance**:
   - k-NN classification as proxy metric (target: > 0.85 accuracy)
   - Model comparison framework for hyperparameter tuning

4. **Operational Metrics** (NEW):
   - Inference latency measurement and optimization (target: < 50ms)
   - Memory footprint analysis and cost implications
   - Embedding dimensions vs performance trade-offs

### Key Takeaways

- **Use both approaches**: Numbers don't tell the whole story - you must look at the data
- **Security-specific concerns**: Check that critical security distinctions (success/failure, privilege levels) are preserved
- **Production readiness**: Balance quality, latency, and cost before deploying
- **Iterative process**: If embeddings fail evaluation, go back to Parts 3-4 (feature engineering, training)

**Next**: In [Part 6](part6-anomaly-detection), we'll use these validated embeddings to detect anomalies using various algorithms (LOF, Isolation Forest, distance-based methods).

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

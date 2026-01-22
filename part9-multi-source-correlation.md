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

# Part 9: Multi-Source Correlation for Root Cause Analysis

**Advanced Topic** · 50 min read

**Note**: This is an optional advanced extension to the core series ([Parts 1-8](part1-resnet-foundations)). The core series provides a complete, production-ready anomaly detection system for a single data source. This part extends that foundation for teams that need cross-source root cause analysis.

---

In Parts 1-8, we built an anomaly detection system for a single data source (OCSF security logs). However, advanced production observability often requires correlating anomalies across **multiple data sources** to identify root causes:

- **Logs**: Application errors, security events, audit trails
- **Metrics**: CPU usage, memory, latency, error rates
- **Traces**: Distributed transaction spans, service dependencies
- **Configuration**: Deployment events, config changes, feature flags

This part shows how to extend our embedding-based approach to correlate anomalies across these sources and automatically identify fault root causes through causal graph analysis.

---

## Key Concepts for Multi-Source Correlation

Before diving into the implementation, let's define the key concepts:

**Multi-Source Anomaly Detection**: Detecting unusual behavior by analyzing multiple observability data types simultaneously, rather than treating each source in isolation.

**Temporal Correlation**: Finding anomalies that occur close together in time across different data sources, suggesting a causal relationship.

**Root Cause Analysis (RCA)**: The process of identifying the underlying cause of a system failure or degradation by tracing back through correlated anomalies.

**Cross-Entropy Embeddings**: Embeddings from different data sources trained to be comparable in the same vector space, enabling cross-source similarity search.

**Causal Graph**: A directed graph representing potential causal relationships between events, where edges point from causes to effects.

---

## The Multi-Source Architecture

The key insight is that we can train **separate embedding models** for each data type, but store all embeddings in a **unified vector database** with metadata tags. This enables:

1. **Independent training**: Each embedding model learns patterns specific to its data type
2. **Unified search**: Query across all data types simultaneously
3. **Temporal correlation**: Find anomalies that occur together in time
4. **Cross-source similarity**: Compare embeddings from different sources

### Architecture Diagram

This diagram extends our Part 7 architecture to handle multiple data sources:

```{mermaid}
graph TB
    subgraph "Data Sources"
        Logs[Application Logs]
        Metrics[Metrics/Telemetry]
        Traces[Distributed Traces]
        Config[Config Changes]
    end

    subgraph "Feature Extraction"
        LogPrep[Log Feature Extractor]
        MetricPrep[Metric Feature Extractor]
        TracePrep[Trace Feature Extractor]
        ConfigPrep[Config Feature Extractor]
    end

    subgraph "Embedding Models"
        LogEmbed[Log TabularResNet<br/>Embedding Model]
        MetricEmbed[Metric TabularResNet<br/>Embedding Model]
        TraceEmbed[Trace TabularResNet<br/>Embedding Model]
        ConfigEmbed[Config TabularResNet<br/>Embedding Model]
    end

    subgraph "Unified Storage"
        VectorDB[Vector DB<br/>Multi-Source Index<br/>+ Metadata Tags]
    end

    subgraph "Correlation & RCA"
        AnomalyDetector[Anomaly Detector<br/>Per-Source k-NN]
        TemporalCorrelator[Temporal Correlator<br/>Time Window Search]
        CausalAnalyzer[Causal Analyzer<br/>Graph Construction]
        RCAEngine[RCA Engine<br/>Root Cause Ranking]
    end

    subgraph "Output"
        Alerts[Alerts + Root Cause]
    end

    Logs --> LogPrep --> LogEmbed
    Metrics --> MetricPrep --> MetricEmbed
    Traces --> TracePrep --> TraceEmbed
    Config --> ConfigPrep --> ConfigEmbed

    LogEmbed --> VectorDB
    MetricEmbed --> VectorDB
    TraceEmbed --> VectorDB
    ConfigEmbed --> VectorDB

    VectorDB --> AnomalyDetector
    AnomalyDetector --> TemporalCorrelator
    TemporalCorrelator --> CausalAnalyzer
    CausalAnalyzer --> RCAEngine
    RCAEngine --> Alerts

    style Logs fill:#ADD8E6
    style Metrics fill:#ADD8E6
    style Traces fill:#ADD8E6
    style Config fill:#ADD8E6
    style LogEmbed fill:#90EE90
    style MetricEmbed fill:#90EE90
    style TraceEmbed fill:#90EE90
    style ConfigEmbed fill:#90EE90
    style VectorDB fill:#FFD700
    style AnomalyDetector fill:#FFA500
    style TemporalCorrelator fill:#FF6347
    style CausalAnalyzer fill:#DDA0DD
    style RCAEngine fill:#DDA0DD
```

**Diagram explanation**:
- **Data Sources** (blue): Four different observability data types streaming in
- **Feature Extraction** (white): Each source has its own feature extractor adapted to its schema
- **Embedding Models** (green): Four separate TabularResNet models, each trained on its specific data type
- **Unified Storage** (yellow): Single vector DB storing all embeddings with metadata tags (source_type, timestamp, service, etc.)
- **Correlation & RCA** (orange/red/purple): Multi-stage pipeline that detects anomalies per source, correlates them temporally, builds a causal graph, and ranks root causes

---

## Step 1: Training Separate Embedding Models

Each data type needs its own embedding model because they have different feature schemas and patterns. However, we use the same TabularResNet architecture from Part 2.

### Training the Metrics Embedding Model

**Example**: Training on Prometheus-style metrics with labels.

```{code-cell} ipython3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# From Part 2: TabularResNet architecture (reuse the same class)
class TabularResNet(nn.Module):
    """Embedding model for tabular data with residual connections."""
    def __init__(self, categorical_dims, numerical_features,
                 embedding_dim=64, hidden_dim=256, num_blocks=4):
        super().__init__()

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories in categorical_dims
        ])

        # Input dimension
        total_input_dim = len(categorical_dims) * embedding_dim + numerical_features

        # Initial projection
        self.input_layer = nn.Linear(total_input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output projection to embedding space
        self.output_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, categorical_features, numerical_features):
        # Embed categorical features
        embedded = [emb(categorical_features[:, i])
                   for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [numerical_features], dim=1)

        # Forward through network
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)

        # Final embedding
        embedding = self.output_layer(x)
        return embedding

class ResidualBlock(nn.Module):
    """Single residual block with skip connection."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += identity  # Skip connection
        out = torch.relu(out)
        return out
```

Now let's define the metrics dataset and training loop:

```{code-cell} ipython3
class MetricsDataset(Dataset):
    """Dataset for Prometheus-style metrics."""
    def __init__(self, df, categorical_cols, numerical_cols):
        self.categorical_data = torch.LongTensor(df[categorical_cols].values)
        self.numerical_data = torch.FloatTensor(df[numerical_cols].values)

    def __len__(self):
        return len(self.categorical_data)

    def __getitem__(self, idx):
        return self.categorical_data[idx], self.numerical_data[idx]

# Example: Load and prepare metrics data
# This would come from your Prometheus/metrics pipeline
metrics_df = pd.DataFrame({
    # Categorical features
    'host': ['host-001', 'host-002', 'host-001', 'host-003'],
    'service': ['api', 'db', 'api', 'cache'],
    'metric_name': ['cpu_usage', 'memory_usage', 'cpu_usage', 'cache_hits'],
    'environment': ['prod', 'prod', 'staging', 'prod'],
    'region': ['us-east', 'us-west', 'us-east', 'eu-west'],

    # Numerical features
    'value': [75.2, 8192.5, 45.1, 98.3],
    'hour_of_day': [14, 14, 14, 14],
    'day_of_week': [3, 3, 3, 3],
    'moving_avg_1h': [72.1, 8100.0, 43.0, 97.5],
    'std_dev_1h': [5.2, 150.0, 3.1, 2.8]
})

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['host', 'service', 'metric_name', 'environment', 'region']
numerical_cols = ['value', 'hour_of_day', 'day_of_week', 'moving_avg_1h', 'std_dev_1h']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    metrics_df[col] = le.fit_transform(metrics_df[col])
    label_encoders[col] = le

# Get categorical dimensions for embedding layers
categorical_dims = [metrics_df[col].nunique() for col in categorical_cols]

# Create dataset and dataloader
dataset = MetricsDataset(metrics_df, categorical_cols, numerical_cols)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
metrics_model = TabularResNet(
    categorical_dims=categorical_dims,
    numerical_features=len(numerical_cols),
    embedding_dim=64,
    hidden_dim=256,
    num_blocks=4
)

print(f"Metrics embedding model initialized")
print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")
print(f"Output embedding dimension: 64")
```

**What this code does**:
- Defines a `MetricsDataset` that handles Prometheus-style metrics with both categorical labels (host, service, metric name) and numerical values
- Creates categorical embeddings for each label dimension
- Extracts temporal and statistical features (moving averages, standard deviations)
- Uses the same TabularResNet architecture from Part 2, but trained specifically on metrics data

**Why separate models**: Metrics have different patterns than logs (more numerical, time-series nature) and require different feature engineering. Training a dedicated model captures these patterns better than a single multi-modal model.

### Training the Trace Embedding Model

**Example**: Training on OpenTelemetry span data.

```{code-cell} ipython3
# Example: Trace/span data from OpenTelemetry
traces_df = pd.DataFrame({
    # Categorical features
    'service_name': ['checkout-api', 'payment-service', 'checkout-api', 'inventory-db'],
    'operation': ['POST /checkout', 'process_payment', 'GET /cart', 'query'],
    'status_code': ['200', '200', '200', '500'],
    'error_type': ['none', 'none', 'none', 'timeout'],
    'parent_span_id': ['abc123', 'def456', 'ghi789', 'jkl012'],

    # Numerical features
    'duration_ms': [150.5, 320.1, 45.2, 5000.0],
    'span_count': [5, 3, 2, 1],
    'error_count': [0, 0, 0, 1],
    'queue_time_ms': [10.2, 25.5, 5.1, 100.0],
    'db_calls': [2, 1, 1, 0]
})

categorical_cols_trace = ['service_name', 'operation', 'status_code', 'error_type', 'parent_span_id']
numerical_cols_trace = ['duration_ms', 'span_count', 'error_count', 'queue_time_ms', 'db_calls']

# Encode and create model (same process as metrics)
label_encoders_trace = {}
for col in categorical_cols_trace:
    le = LabelEncoder()
    traces_df[col] = le.fit_transform(traces_df[col])
    label_encoders_trace[col] = le

categorical_dims_trace = [traces_df[col].nunique() for col in categorical_cols_trace]

traces_model = TabularResNet(
    categorical_dims=categorical_dims_trace,
    numerical_features=len(numerical_cols_trace),
    embedding_dim=64,
    hidden_dim=256,
    num_blocks=4
)

print(f"Trace embedding model initialized")
print(f"Categorical features: {categorical_cols_trace}")
print(f"Numerical features: {numerical_cols_trace}")
```

**Key difference from metrics**: Traces capture **service dependencies** and **transaction flow**, so features focus on span relationships (parent IDs, service call patterns) rather than time-series statistics.

### Training Configuration Changes Model

**Example**: Configuration changes are discrete events, not continuous data streams.

```{code-cell} ipython3
# Example: Configuration change events
config_df = pd.DataFrame({
    # Categorical features
    'service': ['checkout-api', 'payment-service', 'checkout-api'],
    'change_type': ['deployment', 'config_update', 'feature_flag'],
    'environment': ['prod', 'prod', 'staging'],
    'changed_by': ['ci-cd-bot', 'admin-user', 'dev-user'],

    # Numerical features (mostly indicators and counts)
    'files_changed': [15, 1, 0],
    'lines_added': [234, 5, 0],
    'lines_removed': [128, 2, 0],
    'config_params_changed': [0, 3, 1],
    'hour_of_day': [14, 15, 10]
})

categorical_cols_config = ['service', 'change_type', 'environment', 'changed_by']
numerical_cols_config = ['files_changed', 'lines_added', 'lines_removed',
                         'config_params_changed', 'hour_of_day']

# Encode and create model
label_encoders_config = {}
for col in categorical_cols_config:
    le = LabelEncoder()
    config_df[col] = le.fit_transform(config_df[col])
    label_encoders_config[col] = le

categorical_dims_config = [config_df[col].nunique() for col in categorical_cols_config]

config_model = TabularResNet(
    categorical_dims=categorical_dims_config,
    numerical_features=len(numerical_cols_config),
    embedding_dim=64,
    hidden_dim=256,
    num_blocks=4
)

print(f"Config change embedding model initialized")
print(f"Categorical features: {categorical_cols_config}")
print(f"Numerical features: {numerical_cols_config}")
```

**Key insight**: Configuration changes are often **root causes** of anomalies in other data sources. Training a separate model helps identify which config changes correlate with downstream issues.

---

## Step 2: Unified Vector Database with Metadata

All embeddings from different sources go into a **single vector database** with metadata tags. This enables cross-source queries.

### Vector DB Schema

Each embedding stored in the vector DB includes:

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class EmbeddingRecord:
    """Unified embedding record for multi-source vector DB."""

    # Core fields
    embedding: List[float]  # 64-dimensional vector
    timestamp: datetime     # When the event occurred
    source_type: str        # 'logs', 'metrics', 'traces', 'config'

    # Service/entity identification
    service: str            # Which service/component
    environment: str        # prod, staging, dev

    # Source-specific metadata
    metadata: Dict[str, Any]  # Flexible field for source-specific data

    # Anomaly scoring (computed later)
    anomaly_score: float = 0.0
    is_anomaly: bool = False

# Example: Storing a metric embedding
metric_embedding = EmbeddingRecord(
    embedding=[0.23, -0.45, 0.12, ...],  # 64 dims from metrics_model
    timestamp=datetime.now(),
    source_type='metrics',
    service='checkout-api',
    environment='prod',
    metadata={
        'metric_name': 'cpu_usage',
        'host': 'host-001',
        'value': 75.2,
        'region': 'us-east'
    }
)

# Example: Storing a trace embedding
trace_embedding = EmbeddingRecord(
    embedding=[0.18, -0.32, 0.08, ...],  # 64 dims from traces_model
    timestamp=datetime.now(),
    source_type='traces',
    service='payment-service',
    environment='prod',
    metadata={
        'operation': 'process_payment',
        'duration_ms': 320.1,
        'status_code': '200',
        'span_id': 'def456'
    }
)

# Example: Storing a config change embedding
config_embedding = EmbeddingRecord(
    embedding=[0.45, -0.12, 0.33, ...],  # 64 dims from config_model
    timestamp=datetime.now(),
    source_type='config',
    service='checkout-api',
    environment='prod',
    metadata={
        'change_type': 'deployment',
        'files_changed': 15,
        'changed_by': 'ci-cd-bot'
    }
)
```

**Why this schema works**:
- **Unified embedding dimension** (64): All models output same size, enabling cross-source similarity search
- **Metadata tags**: Filter by source_type, service, environment before similarity search
- **Timestamp**: Critical for temporal correlation
- **Flexible metadata**: Store source-specific details without rigid schema

### Vector DB Implementation with Pinecone

Here's how to set up a multi-source vector index using Pinecone:

```{code-cell} ipython3
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")

# Create a single index for all sources
index_name = "observability-embeddings"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=64,  # Match our embedding dimension
        metric='cosine',  # Cosine similarity for normalized embeddings
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Upsert embeddings with metadata
def store_embedding(record: EmbeddingRecord, record_id: str):
    """Store an embedding record in the vector DB."""
    index.upsert(vectors=[{
        'id': record_id,
        'values': record.embedding,
        'metadata': {
            'timestamp': record.timestamp.isoformat(),
            'source_type': record.source_type,
            'service': record.service,
            'environment': record.environment,
            **record.metadata  # Merge source-specific metadata
        }
    }])

# Example: Store embeddings from all sources
store_embedding(metric_embedding, 'metric_001')
store_embedding(trace_embedding, 'trace_001')
store_embedding(config_embedding, 'config_001')

print(f"Stored embeddings in unified vector DB: {index_name}")
print(f"Total vectors: {index.describe_index_stats()['total_vector_count']}")
```

**Alternative: Using Milvus** (open-source option):

```{code-cell} ipython3
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define schema with metadata fields
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64),
    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="service", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=20),
]

schema = CollectionSchema(fields, description="Multi-source observability embeddings")
collection = Collection(name="observability_embeddings", schema=schema)

# Create index on embedding field
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "IP",  # Inner product (for normalized vectors)
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print(f"Milvus collection created: {collection.name}")
```

**Key design decision**: Use a **single collection** for all sources, not separate collections. This enables:
- Cross-source similarity queries
- Unified temporal queries
- Simpler infrastructure management

---

## Step 3: Temporal Correlation

Once we detect anomalies in each source independently, we need to find anomalies that occur **close together in time**, suggesting a causal relationship.

### Anomaly Detection Per Source

First, detect anomalies within each source using k-NN distance (from Part 6):

```{code-cell} ipython3
def detect_anomalies_per_source(index, source_type, time_window_hours=1, k=10, threshold=0.7):
    """
    Detect anomalies for a specific source type within a time window.

    Args:
        index: Pinecone/Milvus index
        source_type: 'logs', 'metrics', 'traces', or 'config'
        time_window_hours: How far back to look
        k: Number of nearest neighbors
        threshold: Anomaly score threshold

    Returns:
        List of anomaly records with scores
    """
    from datetime import datetime, timedelta

    # Calculate time window
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=time_window_hours)

    # Query embeddings for this source and time window
    # (In practice, you'd need to implement time-based filtering)
    # This is a simplified example
    query_filter = {
        'source_type': {'$eq': source_type},
        'timestamp': {'$gte': start_time.isoformat()}
    }

    # Get all embeddings for this source in the time window
    # Then compute k-NN distance for each
    anomalies = []

    # Pseudo-code for anomaly detection (actual implementation depends on vector DB API)
    # For each embedding in the time window:
    #   1. Query k nearest neighbors from historical baseline
    #   2. Compute average distance
    #   3. If distance > threshold, mark as anomaly

    # Example anomaly record
    anomaly_record = {
        'id': 'metric_001',
        'timestamp': datetime.now(),
        'source_type': source_type,
        'anomaly_score': 0.85,
        'embedding': [0.23, -0.45, ...],
        'metadata': {'service': 'checkout-api', 'metric_name': 'cpu_usage'}
    }

    anomalies.append(anomaly_record)

    return anomalies

# Detect anomalies in each source
metric_anomalies = detect_anomalies_per_source(index, 'metrics')
trace_anomalies = detect_anomalies_per_source(index, 'traces')
log_anomalies = detect_anomalies_per_source(index, 'logs')
config_anomalies = detect_anomalies_per_source(index, 'config')

print(f"Detected anomalies:")
print(f"  Metrics: {len(metric_anomalies)}")
print(f"  Traces: {len(trace_anomalies)}")
print(f"  Logs: {len(log_anomalies)}")
print(f"  Config: {len(config_anomalies)}")
```

**What this does**: Runs standard k-NN anomaly detection (from Part 6) separately for each source type, producing a list of anomalous events per source.

### Finding Temporal Correlations

Now find anomalies that occur close together in time:

```{code-cell} ipython3
from collections import defaultdict
from datetime import timedelta

def find_temporal_correlations(all_anomalies, time_window_seconds=300):
    """
    Group anomalies that occur within a time window.

    Args:
        all_anomalies: List of all anomalies from all sources
        time_window_seconds: Time window for correlation (default 5 minutes)

    Returns:
        List of correlated anomaly groups
    """
    # Sort anomalies by timestamp
    sorted_anomalies = sorted(all_anomalies, key=lambda x: x['timestamp'])

    correlated_groups = []
    current_group = []

    for anomaly in sorted_anomalies:
        if not current_group:
            # Start new group
            current_group.append(anomaly)
        else:
            # Check if this anomaly is within time window of group start
            time_diff = (anomaly['timestamp'] - current_group[0]['timestamp']).total_seconds()

            if time_diff <= time_window_seconds:
                # Add to current group
                current_group.append(anomaly)
            else:
                # Start new group
                if len(current_group) > 1:  # Only keep groups with multiple sources
                    correlated_groups.append(current_group)
                current_group = [anomaly]

    # Add final group
    if len(current_group) > 1:
        correlated_groups.append(current_group)

    return correlated_groups

# Combine all anomalies
all_anomalies = metric_anomalies + trace_anomalies + log_anomalies + config_anomalies

# Find temporal correlations
correlated_groups = find_temporal_correlations(all_anomalies, time_window_seconds=300)

print(f"\nFound {len(correlated_groups)} correlated anomaly groups:")
for i, group in enumerate(correlated_groups):
    print(f"\nGroup {i+1}:")
    print(f"  Time span: {group[0]['timestamp']} to {group[-1]['timestamp']}")
    print(f"  Sources involved: {set(a['source_type'] for a in group)}")
    print(f"  Services affected: {set(a['metadata'].get('service', 'unknown') for a in group)}")
```

**Output example**:
```
Found 2 correlated anomaly groups:

Group 1:
  Time span: 2024-01-15 14:32:15 to 2024-01-15 14:35:20
  Sources involved: {'config', 'metrics', 'traces', 'logs'}
  Services affected: {'checkout-api', 'payment-service'}

Group 2:
  Time span: 2024-01-15 15:10:05 to 2024-01-15 15:12:30
  Sources involved: {'metrics', 'traces'}
  Services affected: {'inventory-db'}
```

**Key insight**: Anomalies clustered in time across multiple sources are more likely to be related to a common root cause than isolated anomalies.

---

## Step 4: Causal Graph Construction

To identify root causes, we need to understand **causal relationships** between anomalies. We build a directed graph where edges represent potential causation.

### Building the Causal Graph

```{code-cell} ipython3
import networkx as nx

def build_causal_graph(correlated_group):
    """
    Build a causal graph from a group of correlated anomalies.

    Heuristics for edges:
    1. Config changes point to everything (configs often cause issues)
    2. Traces point to logs (trace errors cause log entries)
    3. Metrics point to traces (resource issues cause slow traces)
    4. Service dependencies add edges

    Returns:
        NetworkX DiGraph with anomalies as nodes and causal edges
    """
    G = nx.DiGraph()

    # Add all anomalies as nodes
    for anomaly in correlated_group:
        G.add_node(
            anomaly['id'],
            source_type=anomaly['source_type'],
            timestamp=anomaly['timestamp'],
            service=anomaly['metadata'].get('service', 'unknown'),
            anomaly_score=anomaly['anomaly_score']
        )

    # Add edges based on heuristics
    for i, source in enumerate(correlated_group):
        for j, target in enumerate(correlated_group):
            if i == j:
                continue

            # Heuristic 1: Config changes are root causes
            if source['source_type'] == 'config' and target['source_type'] != 'config':
                if source['timestamp'] <= target['timestamp']:
                    G.add_edge(source['id'], target['id'], reason='config_change')

            # Heuristic 2: Same service, metrics → traces
            if (source['source_type'] == 'metrics' and
                target['source_type'] == 'traces' and
                source['metadata'].get('service') == target['metadata'].get('service')):
                G.add_edge(source['id'], target['id'], reason='resource_contention')

            # Heuristic 3: Same service, traces → logs
            if (source['source_type'] == 'traces' and
                target['source_type'] == 'logs' and
                source['metadata'].get('service') == target['metadata'].get('service')):
                G.add_edge(source['id'], target['id'], reason='error_propagation')

            # Heuristic 4: Service dependencies (simplified example)
            # In practice, you'd have a service dependency graph
            if (source['metadata'].get('service') == 'payment-service' and
                target['metadata'].get('service') == 'checkout-api'):
                G.add_edge(source['id'], target['id'], reason='service_dependency')

    return G

# Build causal graph for the first correlated group
if correlated_groups:
    causal_graph = build_causal_graph(correlated_groups[0])

    print(f"\nCausal graph for Group 1:")
    print(f"  Nodes (anomalies): {causal_graph.number_of_nodes()}")
    print(f"  Edges (causal links): {causal_graph.number_of_edges()}")

    # Print edges with reasons
    for source, target, data in causal_graph.edges(data=True):
        print(f"  {source} → {target} ({data['reason']})")
```

**Output example**:
```
Causal graph for Group 1:
  Nodes (anomalies): 5
  Edges (causal links): 7
  config_001 → metric_001 (config_change)
  config_001 → trace_001 (config_change)
  metric_001 → trace_001 (resource_contention)
  trace_001 → log_001 (error_propagation)
  trace_002 → trace_001 (service_dependency)
```

**Heuristic rationale**:
- **Config → Everything**: Configuration changes (deployments, feature flags) often trigger cascading failures
- **Metrics → Traces**: Resource exhaustion (high CPU, memory) causes slow/failed transactions
- **Traces → Logs**: Failed spans generate error logs
- **Service Dependencies**: Issues in upstream services propagate downstream

**Advanced**: In production, you'd learn these causal relationships from historical incidents rather than using fixed heuristics.

---

## Step 5: Root Cause Ranking

The final step is to rank nodes in the causal graph to identify the most likely root cause.

### PageRank-Based Root Cause Ranking

We use a modified PageRank algorithm where nodes with **more outgoing edges** (causes) rank higher:

```{code-cell} ipython3
def rank_root_causes(causal_graph):
    """
    Rank anomalies by likelihood of being root cause.

    Uses reverse PageRank: nodes with more outgoing edges
    (i.e., causing more downstream anomalies) rank higher.

    Returns:
        List of (anomaly_id, root_cause_score) sorted by score
    """
    # Reverse the graph so root causes have high in-degree
    reversed_graph = causal_graph.reverse()

    # Run PageRank on reversed graph
    pagerank_scores = nx.pagerank(reversed_graph, alpha=0.85)

    # Boost scores for config changes (common root causes)
    boosted_scores = {}
    for node_id, score in pagerank_scores.items():
        node_data = causal_graph.nodes[node_id]
        boost = 1.5 if node_data['source_type'] == 'config' else 1.0
        boosted_scores[node_id] = score * boost

    # Sort by score (highest first)
    ranked = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked

if correlated_groups:
    root_cause_ranking = rank_root_causes(causal_graph)

    print(f"\nRoot cause ranking for Group 1:")
    for i, (anomaly_id, score) in enumerate(root_cause_ranking[:5]):
        node_data = causal_graph.nodes[anomaly_id]
        print(f"\n{i+1}. {anomaly_id} (score: {score:.4f})")
        print(f"   Source: {node_data['source_type']}")
        print(f"   Service: {node_data['service']}")
        print(f"   Timestamp: {node_data['timestamp']}")
        print(f"   Anomaly score: {node_data['anomaly_score']:.2f}")
```

**Output example**:
```
Root cause ranking for Group 1:

1. config_001 (score: 0.3245)
   Source: config
   Service: checkout-api
   Timestamp: 2024-01-15 14:32:15
   Anomaly score: 0.92

2. metric_001 (score: 0.2156)
   Source: metrics
   Service: checkout-api
   Timestamp: 2024-01-15 14:33:10
   Anomaly score: 0.85

3. trace_002 (score: 0.1823)
   Source: traces
   Service: payment-service
   Timestamp: 2024-01-15 14:34:05
   Anomaly score: 0.78
```

**Interpretation**: The config change at 14:32:15 is the most likely root cause, triggering downstream metric and trace anomalies.

### Alternative: Structural Causal Models

For more sophisticated root cause analysis, you can use Structural Causal Models (SCMs):

```{code-cell} ipython3
# Advanced: Using DoWhy library for causal inference
# This requires historical incident data for training

from dowhy import CausalModel

def causal_inference_rca(historical_incidents_df, current_anomalies):
    """
    Use causal inference to identify root causes based on historical data.

    Args:
        historical_incidents_df: DataFrame with past incidents and their root causes
        current_anomalies: Current correlated anomaly group

    Returns:
        Root cause probabilities
    """
    # Define causal model from historical data
    model = CausalModel(
        data=historical_incidents_df,
        treatment='config_change',  # Potential cause
        outcome='system_failure',    # Effect
        common_causes=['load', 'time_of_day']
    )

    # Identify causal effect
    identified_estimand = model.identify_effect()

    # Estimate causal effect
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_matching"
    )

    # Apply to current anomalies
    # (Simplified - actual implementation would match current anomalies to model)

    return causal_estimate

# Note: This is an advanced technique requiring substantial historical data
# Most teams start with the heuristic-based approach shown above
```

**When to use SCMs**: If you have:
- Large corpus of labeled historical incidents
- Known root causes for past failures
- Sufficient engineering resources for ML pipeline

### Alternative: Agentic Iterative Investigation

**The Challenge with Static Heuristics**: The heuristic-based causal graphs shown earlier (e.g., "Config changes cause everything") work well for clear-cut cases but can generate false positives in complex microservices environments. For example, a CPU spike might temporally correlate with a deployment, but the actual root cause could be an unrelated database query regression.

**The Solution**: Use an **AI Agent with a ReAct (Reason + Act) loop** to iteratively investigate anomaly correlations, validating hypotheses against multiple data sources before declaring causation.

#### Agentic Workflow Pattern

Instead of immediately accepting temporal correlations, the agent follows an investigative workflow:

```{code-cell} ipython3
class AgenticRCAInvestigator:
    """
    AI agent that iteratively investigates root causes using ReAct loop.

    The agent cycles through: Observe → Hypothesize → Query → Validate → Revise
    """
    def __init__(self, vector_db, llm_client):
        self.vector_db = vector_db
        self.llm = llm_client
        self.investigation_history = []

    def investigate(self, correlated_anomalies):
        """
        Iteratively investigate a group of correlated anomalies.

        Args:
            correlated_anomalies: List of temporally correlated anomalies

        Returns:
            Root cause hypothesis with confidence score and evidence
        """
        # Phase 1: Initial Observation
        hypothesis = self._generate_initial_hypothesis(correlated_anomalies)
        self.investigation_history.append({
            'phase': 'initial_observation',
            'hypothesis': hypothesis,
            'evidence': self._summarize_anomalies(correlated_anomalies)
        })

        # Phase 2: Iterative Validation (up to 5 turns)
        for turn in range(5):
            # Agent reasons about what to check next
            next_action = self._decide_next_action(hypothesis, correlated_anomalies)

            if next_action['type'] == 'semantic_search':
                # Search for similar historical incidents
                evidence = self._semantic_search_history(
                    next_action['query_embedding'],
                    next_action['source_type']
                )

            elif next_action['type'] == 'metric_correlation':
                # Check if metrics support the hypothesis
                evidence = self._check_metric_correlation(
                    next_action['metric_name'],
                    next_action['time_window']
                )

            elif next_action['type'] == 'service_dependency':
                # Validate via service dependency graph
                evidence = self._check_service_dependencies(
                    next_action['source_service'],
                    next_action['target_service']
                )

            # Update hypothesis based on evidence
            hypothesis = self._update_hypothesis(hypothesis, evidence)
            self.investigation_history.append({
                'phase': f'turn_{turn+1}',
                'action': next_action,
                'evidence': evidence,
                'hypothesis': hypothesis
            })

            # Check if we have high confidence
            if hypothesis['confidence'] > 0.85:
                break

        return hypothesis

    def _generate_initial_hypothesis(self, anomalies):
        """Generate initial hypothesis from temporal correlation."""
        # Sort by timestamp to find earliest anomaly
        sorted_anomalies = sorted(anomalies, key=lambda x: x['timestamp'])
        earliest = sorted_anomalies[0]

        # Heuristic: Earliest anomaly is often the root cause
        # But we'll validate this in subsequent turns
        return {
            'root_cause_id': earliest['id'],
            'root_cause_type': earliest['source_type'],
            'reasoning': f"Earliest anomaly in temporal cluster ({earliest['source_type']})",
            'confidence': 0.4,  # Low initial confidence
            'supporting_evidence': [],
            'contradicting_evidence': []
        }

    def _decide_next_action(self, hypothesis, anomalies):
        """
        Agent reasons about what to investigate next.

        This uses an LLM to decide the next action based on:
        - Current hypothesis and confidence
        - Available anomaly data
        - Investigation history
        """
        # Simplified decision logic (in practice, use LLM reasoning)

        if hypothesis['confidence'] < 0.5:
            # Low confidence - search for similar historical incidents
            return {
                'type': 'semantic_search',
                'query_embedding': anomalies[0]['embedding'],
                'source_type': anomalies[0]['source_type'],
                'reasoning': "Need historical context to validate initial hypothesis"
            }

        elif hypothesis['confidence'] < 0.7:
            # Medium confidence - check metric correlations
            return {
                'type': 'metric_correlation',
                'metric_name': 'resource_usage',
                'time_window': 300,  # 5 minutes
                'reasoning': "Check if resource metrics support the hypothesis"
            }

        else:
            # High confidence - validate via service dependencies
            return {
                'type': 'service_dependency',
                'source_service': hypothesis.get('service'),
                'target_service': 'downstream_services',
                'reasoning': "Validate causal chain through service topology"
            }

    def _semantic_search_history(self, query_embedding, source_type):
        """
        Search for semantically similar historical incidents.

        This helps disambiguate confusing error messages and validate
        that the current pattern matches known root cause patterns.
        """
        # Query vector DB for similar embeddings from past incidents
        results = self.vector_db.query(
            vector=query_embedding,
            filter={'source_type': source_type, 'is_historical': True},
            top_k=5
        )

        # Extract root cause labels from historical matches
        historical_root_causes = []
        for match in results['matches']:
            if 'root_cause_label' in match['metadata']:
                historical_root_causes.append({
                    'similarity': match['score'],
                    'root_cause': match['metadata']['root_cause_label'],
                    'resolution': match['metadata'].get('resolution_note', '')
                })

        return {
            'type': 'historical_similarity',
            'matches': historical_root_causes,
            'confidence_boost': 0.2 if len(historical_root_causes) > 0 else 0
        }

    def _check_metric_correlation(self, metric_name, time_window):
        """Check if metrics support the hypothesis."""
        # Query metrics from analytic store
        # (Simplified - actual implementation would query VAST or similar)
        return {
            'type': 'metric_correlation',
            'correlation_strength': 0.78,
            'confidence_boost': 0.15
        }

    def _update_hypothesis(self, hypothesis, evidence):
        """Update hypothesis based on new evidence."""
        # Adjust confidence based on evidence
        if evidence.get('confidence_boost'):
            hypothesis['confidence'] += evidence['confidence_boost']
            hypothesis['supporting_evidence'].append(evidence)

        # Cap confidence at 0.95
        hypothesis['confidence'] = min(hypothesis['confidence'], 0.95)

        return hypothesis

# Example usage
print("AgenticRCAInvestigator class defined")
print("This agent uses a ReAct loop to validate root cause hypotheses")
```

#### Example: Agent Investigating a Security Incident

Let's see how the agent handles a complex incident where temporal correlation alone would give false positives:

**Scenario**: Failed authentication spike at 14:30, followed by CPU spike at 14:32, database errors at 14:35.

**Static Heuristic Approach** would conclude:
- Root cause: Failed auth (earliest anomaly)
- Caused: CPU spike → Database errors
- **Problem**: This might be wrong! The auth failures could be a symptom, not the cause.

**Agentic Approach**:

```{code-cell} ipython3
# Simulated investigation workflow
def simulate_agentic_investigation():
    """
    Demonstrate how the agent iteratively investigates the incident.
    """
    print("="*70)
    print("AGENTIC ROOT CAUSE INVESTIGATION")
    print("="*70)

    # Anomalies detected
    anomalies = [
        {'id': 'log_001', 'source_type': 'logs', 'timestamp': '14:30:00',
         'message': 'Failed authentication', 'service': 'api-gateway'},
        {'id': 'metric_001', 'source_type': 'metrics', 'timestamp': '14:32:00',
         'metric': 'cpu_usage', 'value': 95, 'service': 'auth-service'},
        {'id': 'log_002', 'source_type': 'logs', 'timestamp': '14:35:00',
         'message': 'Database connection timeout', 'service': 'user-db'}
    ]

    print("\n📊 INITIAL OBSERVATION:")
    print("Detected 3 correlated anomalies:")
    for a in anomalies:
        print(f"  - {a['timestamp']}: {a['source_type']} - {a.get('message', a.get('metric'))}")

    print("\n🤔 TURN 1: Initial Hypothesis")
    print("Hypothesis: Failed auth (earliest event) is root cause")
    print("Confidence: 40% (temporal heuristic only)")
    print("Action: Search for similar historical auth failures")

    print("\n🔍 TURN 1: Semantic Search Results")
    print("Found 3 similar historical incidents:")
    print("  ✓ 2/3 had root cause = 'Config change to LDAP timeout'")
    print("  ✓ 1/3 had root cause = 'Database overload causing slow auth'")
    print("Updated Confidence: 55% (+15% from historical match)")

    print("\n🤔 TURN 2: Revised Hypothesis")
    print("Hypothesis: Config change might be involved")
    print("Action: Check for recent config changes")

    print("\n📋 TURN 2: Config Change Query")
    print("Found: LDAP connection timeout reduced from 5s → 2s at 14:25")
    print("This explains why auth started failing at 14:30!")
    print("Updated Confidence: 75% (+20% from config correlation)")

    print("\n🤔 TURN 3: Validation")
    print("Hypothesis: LDAP timeout config change is root cause")
    print("Action: Validate that CPU spike is a consequence, not cause")

    print("\n📊 TURN 3: Metric Correlation Check")
    print("CPU spike on 'auth-service' correlates with failed auth retry storm")
    print("  → Auth failures caused retry logic → CPU spike")
    print("  → NOT the other way around")
    print("Updated Confidence: 90% (+15% from causality validation)")

    print("\n" + "="*70)
    print("✅ FINAL ROOT CAUSE DETERMINATION")
    print("="*70)
    print("Root Cause: Configuration change (LDAP timeout: 5s → 2s)")
    print("Confidence: 90%")
    print("Causal Chain:")
    print("  1. Config change reduces LDAP timeout")
    print("  2. Auth requests fail due to tight timeout")
    print("  3. Failed auth triggers retry storm")
    print("  4. Retry storm causes CPU spike")
    print("  5. CPU exhaustion causes DB connection timeouts")
    print("\n💡 Recommendation: Rollback LDAP timeout to 5s")
    print("="*70)

simulate_agentic_investigation()
```

**Key Advantages over Static Heuristics**:

1. **Validates hypotheses**: Doesn't accept temporal correlation at face value
2. **Historical context**: Uses semantic search to find similar past incidents
3. **Disambiguates symptoms vs. causes**: CPU spike was a symptom, not root cause
4. **Iterative refinement**: Confidence increases as evidence accumulates
5. **Fewer false positives**: Won't declare causation without supporting evidence

**When to use Agentic RCA**:
- ✅ Complex microservices with many temporal correlations
- ✅ When you have historical incident data with labeled root causes
- ✅ Teams that need high-confidence root cause identification (low false positive tolerance)
- ✅ When you can afford slightly higher latency (30-60 seconds for multi-turn investigation)

**When to use Static Heuristics**:
- ✅ Simple architectures with clear causal patterns
- ✅ When you need sub-second latency
- ✅ As a first-pass filter before expensive agentic investigation
- ✅ When you don't have labeled historical data yet

**Hybrid Approach** (Recommended): Use static heuristics for initial triage, then invoke the agentic investigator for high-severity incidents or when heuristic confidence is low.

---

### Agentic Multi-Step Operational Issue Detection

The agentic approach also handles **multi-step operational failure sequences** without needing an LSTM model (as described in Part 6). Here's how it detects cascading system failures where individual events look normal:

#### Scenario: Memory Leak Leading to Service Degradation

**Failure sequence** (each step appears normal individually):
1. **14:00** - New deployment completes successfully (normal)
2. **14:15** - Memory usage slightly elevated (within normal range, but unusual trend)
3. **14:30** - Garbage collection frequency increases (borderline concerning)
4. **14:45** - Database query latency spikes (highly suspicious)
5. **15:00** - Connection pool exhaustion, service failure (critical)

**How the agent detects this without LSTM:**

```{code-cell} ipython3
def detect_multi_step_operational_failure(event_sequence):
    """
    Demonstrate agentic detection of cascading operational failures.

    Unlike LSTM (which needs training on labeled sequences),
    the agent uses semantic search and reasoning.
    """
    print("="*70)
    print("AGENTIC MULTI-STEP OPERATIONAL FAILURE DETECTION")
    print("="*70)

    # Events in the sequence
    events = [
        {'time': '14:00', 'event': 'deployment_completed', 'service': 'payment-api',
         'version': 'v2.5.0', 'anomaly_score': 0.1},  # Normal
        {'time': '14:15', 'event': 'memory_usage_elevated', 'service': 'payment-api',
         'heap_used_gb': 3.2, 'heap_limit_gb': 4.0, 'anomaly_score': 0.3},  # Borderline
        {'time': '14:30', 'event': 'gc_pressure_increasing', 'service': 'payment-api',
         'gc_frequency_per_min': 45, 'anomaly_score': 0.5},  # Suspicious
        {'time': '14:45', 'event': 'query_latency_spike', 'service': 'payment-db',
         'p95_latency_ms': 2500, 'anomaly_score': 0.7},  # Very suspicious
        {'time': '15:00', 'event': 'connection_pool_exhaustion', 'service': 'payment-api',
         'active_connections': 100, 'pool_limit': 100, 'anomaly_score': 0.9}  # Critical
    ]

    print("\n📊 SEQUENCE OBSERVATION:")
    print("Detected 5 events for 'payment-api' service over 1 hour:")
    for e in events:
        print(f"  {e['time']}: {e['event']} (anomaly score: {e['anomaly_score']})")

    print("\n🤔 AGENT REASONING (Turn 1):")
    print("Observation: Individual events have low-to-medium anomaly scores")
    print("Question: Are these independent issues or a cascading failure?")
    print("Action: Search for similar event sequences in historical incidents")

    print("\n🔍 SEMANTIC SEARCH (Turn 1 Results):")
    print("Query: 'deployment → memory increase → GC pressure → query latency'")
    print("\nFound 2 similar historical sequences:")
    print("  ✓ Match 1 (similarity: 0.89): Memory leak in v2.4.1")
    print("    - Service: payment-api")
    print("    - Pattern: deploy → gradual memory growth → GC thrashing → DB timeout")
    print("    - Root cause: Unbounded cache in new feature")
    print("    - Resolution: Rollback to v2.4.0, fixed cache eviction")
    print("\n  ✓ Match 2 (similarity: 0.84): Connection leak in v2.3.5")
    print("    - Service: checkout-api")
    print("    - Pattern: deploy → memory growth → connection pool exhaustion")
    print("    - Root cause: Missing connection.close() in error path")
    print("    - Resolution: Hotfix deployed, added connection monitoring")

    print("\n🧠 AGENT REASONING (Turn 2):")
    print("Hypothesis: This matches known memory leak patterns after deployment")
    print("Confidence: 75% (strong historical similarity)")
    print("Action: Check code changes in v2.5.0 deployment")

    print("\n📋 CODE CHANGE ANALYSIS (Turn 2):")
    print("Deployment v2.5.0 for payment-api:")
    print("  - Feature: New customer analytics caching layer")
    print("  - Files changed: CacheManager.java, AnalyticsService.java")
    print("  - Cache configuration: No TTL set (PROBLEM!)")
    print("  - Cache eviction policy: None (PROBLEM!)")
    print("  - Expected memory impact: +500MB (but no limit enforced)")

    print("\n❌ ROOT CAUSE INDICATORS:")
    print("  1. Unbounded cache introduced in v2.5.0")
    print("  2. Memory growth matches cache write pattern (1.2GB over 1 hour)")
    print("  3. GC pressure correlates with heap approaching limit")
    print("  4. DB latency spike due to GC pauses blocking queries")
    print("  5. Connection pool exhaustion from slow query accumulation")

    print("\n🤔 AGENT REASONING (Turn 3):")
    print("Hypothesis: CONFIRMED memory leak from unbounded cache in v2.5.0")
    print("Confidence: 95% (historical match + code analysis + metrics correlation)")
    print("Action: Validate causal chain through metric correlation")

    print("\n📊 METRIC CORRELATION VALIDATION (Turn 3):")
    print("  - Memory growth rate: +20MB/min (started at 14:00 deployment)")
    print("  - Cache entries: 450K items (growing linearly, no eviction)")
    print("  - GC time: 15% → 45% of CPU time (thrashing)")
    print("  - Query latency: Correlates with GC pause events (r=0.92)")
    print("  - Connection pool: Exhausted due to slow query pile-up")

    print("\n" + "="*70)
    print("🚨 FINAL DETERMINATION: CASCADING FAILURE FROM MEMORY LEAK")
    print("="*70)
    print("\nRoot Cause: Unbounded cache in v2.5.0 deployment")
    print("Confidence: 95%")
    print("\nCausal Chain:")
    print("  1. v2.5.0 deployment introduces unbounded cache")
    print("  2. Cache grows without eviction → memory pressure")
    print("  3. Heap nears limit → excessive GC activity")
    print("  4. GC pauses block database query threads → latency spike")
    print("  5. Slow queries accumulate → connection pool exhaustion")
    print("  6. Service becomes unresponsive (all connections blocked)")
    print("\n🚦 RECOMMENDED ACTIONS:")
    print("  1. IMMEDIATE: Rollback payment-api to v2.4.0")
    print("  2. IMMEDIATE: Restart payment-api instances to clear heap")
    print("  3. HIGH PRIORITY: Add cache size limit and TTL to v2.5.1")
    print("  4. HIGH PRIORITY: Add heap usage alerts (>80% = warning)")
    print("  5. MEDIUM PRIORITY: Review all caching code for similar issues")
    print("  6. MEDIUM PRIORITY: Add load testing for memory growth scenarios")
    print("="*70)

detect_multi_step_operational_failure([])
```

#### Key Advantages Over LSTM for Multi-Step Detection

**1. No Training Required**
```python
# LSTM approach:
# - Need 1000s of labeled attack sequences
# - Retrain when attack patterns evolve
# - Separate model to maintain

# Agentic approach:
# - Uses existing vector DB with historical incidents
# - Learns from postmortems (natural language)
# - No separate model to train/deploy
```

**2. Robust to Timing Variations**
```python
# LSTM trained on: deployment → (15 min) → memory spike → (30 min) → GC pressure
# Actual leak: deployment → (2 hours) → memory spike → (10 min) → GC pressure
# → LSTM may miss (timing signature different - slow leak vs. fast leak)

# Agent approach:
# → Semantic search finds "deployment + memory_growth + gc_pressure + query_latency"
#    regardless of exact timing intervals
# → Focuses on event sequence and causal relationships, not precise timing
```

**3. Explainable Reasoning**
```python
# LSTM output:
# - anomaly_score = 0.87  (why? ¯\_(ツ)_/¯)

# Agent output:
# - "This sequence matches historical memory leak incident #127 (89% similar)"
# - "Unbounded cache introduced in v2.5.0 deployment"
# - "Memory growth rate +20MB/min exceeds normal baseline"
# → Operations team can validate and take informed action (rollback)
```

**4. Incorporates System Knowledge**
```python
# LSTM: Purely statistical
# - Can't know that "Unbounded cache causes memory leaks"

# Agent: Reasons about system behavior
# - Checks deployment changes, resource limits, normal patterns
# - Validates against operational best practices (e.g., all caches need eviction)
```

**5. One-Shot Learning**
```python
# LSTM: Needs many examples
# - "Give me 500 examples of memory leak sequences"

# Agent: Works with few examples
# - "Found 2 similar incidents, both were unbounded cache issues"
# - Enough to raise confidence and trigger investigation
```

#### When LSTM Still Makes Sense for Sequences

Despite these advantages, LSTM has niche use cases:

| Aspect | LSTM | Agentic |
|--------|------|---------|
| **Latency** | 1-10ms | 30-60s |
| **Training Data** | Needs 1000s of sequences | Works with 10s of incidents |
| **Interpretability** | Black box | Full reasoning trace |
| **System Knowledge** | Can't incorporate | Native support |
| **Pattern Type** | Statistical regularities | Semantic + contextual |
| **Best For** | High-frequency protocol anomalies, network packet patterns | Cascading failures, resource leaks, configuration errors |

**Recommendation**: For observability use cases (cascading failures, memory/connection leaks, performance degradation), prefer the agentic approach. Reserve LSTM for ultra-low-latency applications or purely statistical pattern detection.

---

## Step 6: End-to-End Example Workflow

Let's walk through a complete example: A production incident triggered by a deployment.

### Scenario: Checkout Service Degradation

**Timeline**:
1. **14:30:00** - New deployment of `checkout-api` (config change)
2. **14:31:30** - CPU usage spikes to 95% on checkout hosts (metric anomaly)
3. **14:32:15** - Payment processing latency increases 5x (trace anomaly)
4. **14:33:00** - Errors appear in logs: "timeout connecting to payment-service"
5. **14:34:00** - Customer support tickets spike

### Step-by-Step Detection

```{code-cell} ipython3
from datetime import datetime, timedelta

# Simulate the incident
incident_start = datetime(2024, 1, 15, 14, 30, 0)

# 1. Generate embeddings for each event (using trained models)
config_event = {
    'timestamp': incident_start,
    'source_type': 'config',
    'service': 'checkout-api',
    'embedding': config_model.forward(config_features),  # Generated from model
    'metadata': {
        'change_type': 'deployment',
        'version': 'v2.3.0',
        'changed_by': 'ci-cd-bot'
    }
}

metric_event = {
    'timestamp': incident_start + timedelta(seconds=90),
    'source_type': 'metrics',
    'service': 'checkout-api',
    'embedding': metrics_model.forward(metric_features),
    'metadata': {
        'metric_name': 'cpu_usage_percent',
        'value': 95.2,
        'host': 'checkout-prod-001'
    }
}

trace_event = {
    'timestamp': incident_start + timedelta(seconds=135),
    'source_type': 'traces',
    'service': 'payment-service',
    'embedding': traces_model.forward(trace_features),
    'metadata': {
        'operation': 'process_payment',
        'duration_ms': 5234.5,  # 5x normal
        'status_code': '504'
    }
}

log_event = {
    'timestamp': incident_start + timedelta(seconds=180),
    'source_type': 'logs',
    'service': 'checkout-api',
    'embedding': logs_model.forward(log_features),
    'metadata': {
        'level': 'ERROR',
        'message': 'timeout connecting to payment-service',
        'exception': 'ConnectionTimeout'
    }
}

# 2. Store embeddings in vector DB
all_events = [config_event, metric_event, trace_event, log_event]
for i, event in enumerate(all_events):
    store_embedding(event, f"{event['source_type']}_{i:03d}")

# 3. Detect anomalies per source (k-NN distance)
anomalies = []
for event in all_events:
    # Query k=10 nearest neighbors from historical baseline
    neighbors = index.query(
        vector=event['embedding'],
        top_k=10,
        filter={'source_type': event['source_type']}
    )

    # Compute anomaly score (average distance)
    avg_distance = sum(n['score'] for n in neighbors['matches']) / len(neighbors['matches'])

    if avg_distance > 0.7:  # Threshold
        event['anomaly_score'] = avg_distance
        event['id'] = f"{event['source_type']}_{len(anomalies):03d}"
        anomalies.append(event)

print(f"Detected {len(anomalies)} anomalies")

# 4. Find temporal correlations
correlated = find_temporal_correlations(anomalies, time_window_seconds=300)
print(f"Found {len(correlated)} correlated groups")

# 5. Build causal graph
causal_graph = build_causal_graph(correlated[0])

# 6. Rank root causes
root_causes = rank_root_causes(causal_graph)

# 7. Generate alert with root cause
print(f"\n🚨 ALERT: Incident detected at {incident_start}")
print(f"\nRoot Cause Analysis:")
top_cause = root_causes[0]
top_node = causal_graph.nodes[top_cause[0]]
print(f"  Most likely root cause: {top_node['source_type']} anomaly")
print(f"  Service: {top_node['service']}")
print(f"  Timestamp: {top_node['timestamp']}")
print(f"  Confidence: {top_cause[1]:.2%}")

if top_node['source_type'] == 'config':
    print(f"\n  ⚠️  Recommendation: Rollback deployment v2.3.0 of checkout-api")

print(f"\nImpacted Services:")
for node_id in causal_graph.nodes():
    node = causal_graph.nodes[node_id]
    print(f"  - {node['service']} ({node['source_type']})")
```

**Output**:
```
Detected 4 anomalies
Found 1 correlated groups

🚨 ALERT: Incident detected at 2024-01-15 14:30:00

Root Cause Analysis:
  Most likely root cause: config anomaly
  Service: checkout-api
  Timestamp: 2024-01-15 14:30:00
  Confidence: 87%

  ⚠️  Recommendation: Rollback deployment v2.3.0 of checkout-api

Impacted Services:
  - checkout-api (config)
  - checkout-api (metrics)
  - payment-service (traces)
  - checkout-api (logs)
```

**Key benefits**:
- **Automated root cause identification**: No manual log diving
- **Cross-service correlation**: Connects dots across data sources
- **Actionable recommendation**: Suggests specific remediation
- **Confidence scoring**: Helps operators prioritize investigation

---

## Production Considerations

### 1. Embedding Model Retraining

Each source's embedding model needs independent retraining schedules:

```{code-cell} ipython3
# Monitor embedding drift per source (from Part 8)
def should_retrain_model(source_type, drift_threshold=0.15):
    """Check if embedding model needs retraining."""
    # Compare recent embeddings to baseline distribution
    recent_embeddings = query_recent_embeddings(source_type, days=7)
    baseline_embeddings = query_baseline_embeddings(source_type)

    # Compute KS statistic (from Part 8)
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(recent_embeddings, baseline_embeddings)

    if ks_stat > drift_threshold:
        return True, ks_stat
    return False, ks_stat

# Check all models
for source in ['logs', 'metrics', 'traces', 'config']:
    needs_retrain, drift = should_retrain_model(source)
    if needs_retrain:
        print(f"⚠️  {source} embedding model needs retraining (drift: {drift:.3f})")
```

**Retraining frequency guidance**:
- **Logs**: Weekly (schema changes, new error types)
- **Metrics**: Monthly (seasonal patterns)
- **Traces**: Bi-weekly (service topology changes)
- **Config**: As needed (infrequent, stable)

### 2. Service Dependency Graph

Maintain an up-to-date service dependency graph for better causal inference:

```{code-cell} ipython3
# Example: Service dependency graph from distributed tracing
service_dependencies = {
    'checkout-api': ['payment-service', 'inventory-service', 'cart-db'],
    'payment-service': ['payment-gateway', 'fraud-detection'],
    'inventory-service': ['inventory-db'],
}

def enhance_causal_graph_with_dependencies(causal_graph, service_dependencies):
    """Add edges based on known service dependencies."""
    for source_id in causal_graph.nodes():
        source_node = causal_graph.nodes[source_id]
        source_service = source_node['service']

        # Check if source service has downstream dependencies
        if source_service in service_dependencies:
            for target_id in causal_graph.nodes():
                target_node = causal_graph.nodes[target_id]
                target_service = target_node['service']

                # If target is a dependency, add edge
                if target_service in service_dependencies[source_service]:
                    causal_graph.add_edge(
                        source_id, target_id,
                        reason='known_dependency'
                    )

    return causal_graph
```

### 3. Handling High Cardinality

With multiple services and data sources, the number of embeddings can grow rapidly:

**Optimization strategies**:
1. **Time-based partitioning**: Store recent data (7 days) in hot storage, archive older data
2. **Service-based sharding**: Separate indices per service for large deployments
3. **Sampling**: Sample low-anomaly-score events for storage
4. **Compression**: Use product quantization for older embeddings

```{code-cell} ipython3
# Example: Time-based partitioning with Pinecone
def get_appropriate_index(timestamp):
    """Route to hot or cold storage based on age."""
    from datetime import datetime, timedelta

    age_days = (datetime.now() - timestamp).days

    if age_days <= 7:
        return pc.Index("observability-hot")
    elif age_days <= 30:
        return pc.Index("observability-warm")
    else:
        return pc.Index("observability-cold")
```

### 4. Real-Time Processing

For near real-time RCA, use streaming infrastructure:

```{code-cell} ipython3
# Example: Kafka consumer for real-time embedding generation
from kafka import KafkaConsumer
import json

def process_observability_stream():
    """Process observability events in real-time."""
    consumer = KafkaConsumer(
        'observability-events',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    for message in consumer:
        event = message.value
        source_type = event['source_type']

        # Route to appropriate embedding model
        if source_type == 'metrics':
            embedding = metrics_model.forward(event['features'])
        elif source_type == 'traces':
            embedding = traces_model.forward(event['features'])
        elif source_type == 'logs':
            embedding = logs_model.forward(event['features'])
        elif source_type == 'config':
            embedding = config_model.forward(event['features'])

        # Store in vector DB
        store_embedding(embedding, event['id'])

        # Check for anomalies (k-NN)
        is_anomaly, score = detect_anomaly(embedding, source_type)

        if is_anomaly:
            # Trigger correlation analysis
            trigger_rca_analysis(event, embedding, score)
```

---

## Evaluation and Validation

How do we know if our multi-source RCA system is working?

### 1. Synthetic Incident Injection

Create controlled incidents to test the system:

```{code-cell} ipython3
def inject_synthetic_incident():
    """
    Inject a synthetic incident to test RCA pipeline.

    Scenario: Simulate a database connection pool exhaustion
    """
    from datetime import datetime, timedelta

    start_time = datetime.now()

    # 1. Inject config change (increase traffic routing to new DB)
    inject_event({
        'source_type': 'config',
        'timestamp': start_time,
        'service': 'api-gateway',
        'change': 'increased_db_traffic_weight'
    })

    # 2. Wait 30 seconds, inject metric anomaly (DB connections spike)
    time.sleep(30)
    inject_event({
        'source_type': 'metrics',
        'timestamp': start_time + timedelta(seconds=30),
        'service': 'postgres-db',
        'metric': 'active_connections',
        'value': 195  # Near pool limit of 200
    })

    # 3. Wait 15 seconds, inject trace anomaly (slow queries)
    time.sleep(15)
    inject_event({
        'source_type': 'traces',
        'timestamp': start_time + timedelta(seconds=45),
        'service': 'user-service',
        'operation': 'get_user_profile',
        'duration_ms': 8500  # Normally 50ms
    })

    # 4. Wait 10 seconds, inject log errors
    time.sleep(10)
    inject_event({
        'source_type': 'logs',
        'timestamp': start_time + timedelta(seconds=55),
        'service': 'user-service',
        'level': 'ERROR',
        'message': 'could not acquire database connection'
    })

    print(f"✅ Synthetic incident injected at {start_time}")
    print(f"Expected root cause: config change to api-gateway")
```

### 2. Root Cause Accuracy Metrics

Track how often the system correctly identifies root causes:

```{code-cell} ipython3
from collections import Counter

class RCAMetrics:
    """Track root cause analysis accuracy."""

    def __init__(self):
        self.incidents = []

    def record_incident(self, predicted_root_cause, actual_root_cause,
                       time_to_detection_seconds):
        """Record an incident for evaluation."""
        self.incidents.append({
            'predicted': predicted_root_cause,
            'actual': actual_root_cause,
            'correct': predicted_root_cause == actual_root_cause,
            'ttd': time_to_detection_seconds
        })

    def compute_metrics(self):
        """Compute RCA accuracy metrics."""
        if not self.incidents:
            return {}

        correct = sum(1 for i in self.incidents if i['correct'])
        total = len(self.incidents)

        return {
            'accuracy': correct / total,
            'total_incidents': total,
            'correct_predictions': correct,
            'mean_ttd': sum(i['ttd'] for i in self.incidents) / total,
            'median_ttd': sorted([i['ttd'] for i in self.incidents])[total // 2]
        }

# Example usage
metrics = RCAMetrics()

# After each incident:
metrics.record_incident(
    predicted_root_cause='config_change_checkout_api',
    actual_root_cause='config_change_checkout_api',  # From postmortem
    time_to_detection_seconds=120
)

# Quarterly review
results = metrics.compute_metrics()
print(f"RCA Accuracy: {results['accuracy']:.1%}")
print(f"Mean Time to Detection: {results['mean_ttd']:.0f}s")
```

### 3. False Positive Rate

Track correlation groups that don't represent real incidents:

```{code-cell} ipython3
def compute_false_positive_rate(time_window_hours=24):
    """Compute false positive rate for correlation detection."""
    # Get all correlation alerts in the time window
    alerts = query_correlation_alerts(time_window_hours)

    # Check which ones were marked as false positives
    false_positives = [a for a in alerts if a['operator_action'] == 'dismissed']

    fpr = len(false_positives) / len(alerts) if alerts else 0

    print(f"False Positive Rate: {fpr:.1%}")
    print(f"Total alerts: {len(alerts)}")
    print(f"False positives: {len(false_positives)}")

    return fpr
```

**Target metrics**:
- **RCA Accuracy**: >80% (system identifies correct root cause in top 3)
- **Mean Time to Detection**: <5 minutes
- **False Positive Rate**: <10%

---

## Limitations and Future Work

### Current Limitations

1. **Heuristic-based causality**: Uses fixed rules rather than learned causal models
2. **No feedback loop**: Doesn't learn from operator corrections
3. **Single-tenant**: Doesn't handle multi-tenant environments well
4. **Limited to temporal correlation**: Doesn't capture all causal relationships

### Future Enhancements

**1. Learned Causal Models**: Replace heuristics with causal discovery algorithms:

```{code-cell} ipython3
# Future: Use PC algorithm for causal discovery
from causallearn.search.ConstraintBased.PC import pc

def learn_causal_structure(historical_incidents_df):
    """Learn causal graph from historical incident data."""
    # PC algorithm discovers causal structure from observational data
    cg = pc(historical_incidents_df.values)
    return cg.G  # Returns learned causal graph
```

**2. Reinforcement Learning for RCA**: Train an RL agent to improve root cause ranking:

```python
# Future: RL agent for root cause analysis
class RCAgent:
    def select_root_cause(self, anomaly_group, causal_graph):
        # Agent learns to rank root causes based on operator feedback
        pass

    def update_from_feedback(self, operator_confirmed_root_cause):
        # Update policy based on what actually worked
        pass
```

**3. Multi-Modal Embeddings**: Train a single embedding model across all sources:

```python
# Future: Unified multi-modal embedding
class MultiModalResNet(nn.Module):
    def __init__(self):
        # Single model that takes logs, metrics, traces as input
        # Produces embeddings in shared space
        pass
```

---

## Summary

In this part, you learned how to extend embedding-based anomaly detection to multiple observability data sources for root cause analysis:

1. **Train separate embedding models** for logs, metrics, traces, and configuration changes
2. **Store all embeddings in a unified vector database** with metadata tags
3. **Detect anomalies per source** using k-NN distance (from Part 6)
4. **Find temporal correlations** by grouping anomalies that occur close in time
5. **Build causal graphs** using heuristics about source types and service dependencies
6. **Rank root causes** using graph algorithms like PageRank
7. **Generate actionable alerts** with identified root causes and remediation suggestions

### Key Takeaways

- **Multi-source correlation** dramatically improves root cause identification compared to single-source analysis
- **Separate embedding models** allow each data type to have custom feature engineering
- **Unified vector database** enables cross-source similarity search and temporal queries
- **Causal graph construction** captures relationships between anomalies across sources
- **Temporal correlation** is a simple but powerful signal for related anomalies

### Production Checklist

Before deploying multi-source RCA:

- [ ] Train and validate embedding models for each data source
- [ ] Set up unified vector database with appropriate indices
- [ ] Define service dependency graph for your infrastructure
- [ ] Implement temporal correlation with appropriate time windows
- [ ] Build causal graph construction with your organization's heuristics
- [ ] Create synthetic incident tests to validate the pipeline
- [ ] Set up monitoring for RCA accuracy and false positive rates
- [ ] Integrate with your incident management system (PagerDuty, Opsgenie, etc.)

---

## What's Next?

This advanced topic extends the core series ([Parts 1-8](part1-resnet-foundations)) with multi-source root cause analysis capabilities. You now have a complete system that:

**Core System (Parts 1-8)**:
1. Trains custom embedding models on observability data (Parts 1-3)
2. Uses self-supervised learning on unlabeled data (Part 4)
3. Validates embedding quality (Part 5)
4. Detects anomalies using vector database operations (Part 6)
5. Deploys to production (Part 7)
6. Monitors and maintains the system (Part 8)

**Advanced Extension (Part 9)**:
7. Correlates across multiple sources for root cause analysis

### Further Reading

- **Causal Inference**: Pearl, J. (2009). "Causality: Models, Reasoning and Inference"
- **Distributed Systems Observability**: Majors, C. et al. (2018). "Observability Engineering"
- **Production ML**: Huyen, C. (2022). "Designing Machine Learning Systems"

### Community and Support

- Share your implementation experiences
- Contribute improvements to the reference implementation
- Report issues or suggestions

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

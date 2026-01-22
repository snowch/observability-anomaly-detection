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

# Part 7: Production Deployment

Deploy your anomaly detection system to production with REST APIs for embedding model serving and integration with observability platforms.

**What you'll learn**: How to package your trained custom embedding model (TabularResNet) as a production service that processes OCSF events in near real-time. For each incoming event, the service: (1) generates an embedding using TabularResNet, (2) queries the vector DB for k-nearest neighbors via similarity search, and (3) computes an anomaly score from the neighbor distances. This entire pipeline runs in milliseconds per event.

## Key Technologies

We'll use industry-standard tools for production ML deployment:

- **FastAPI**: Modern Python web framework for building REST APIs. Fast, automatic documentation, type-safe.
- **Docker**: Containerization platform to package your service with all dependencies for consistent deployment.
- **MLflow**: Open-source platform for model versioning, experiment tracking, and model registry.
- **Prometheus/Grafana**: Monitoring stack for tracking system metrics and performance.
- **Kafka**: Distributed streaming platform for handling OCSF event streams at scale.

---

## Deployment Architecture

### System Components

The diagram below shows the complete production architecture. For each OCSF event:
1. **Kafka** streams the event to the preprocessor
2. **Preprocessor** extracts and encodes features
3. **Embedding Service** generates a 64-dim vector using TabularResNet
4. **Vector DB** performs k-NN similarity search against historical embeddings
5. **Anomaly Scoring** computes distance-based anomaly score (pure vector operations)
6. **Alert Manager** triggers alerts for high-scoring anomalies

The model registry enables versioning and rollbacks of the embedding model only (no classical detection model).

```{mermaid}
graph TB
    OCSF[OCSF Data Stream<br/>Kafka] --> Preprocessor[Preprocessor Service<br/>Feature Engineering]
    Preprocessor --> Embedding[Embedding Service<br/>TabularResNet]
    Embedding --> VectorDB[Vector DB<br/>Index + Similarity Search]
    VectorDB --> Detector[Anomaly Scoring<br/>k-NN Distance/Density]
    Detector --> AlertManager[Alert Manager]
    AlertManager --> Observability[Observability Platform<br/>Prometheus/Grafana]

    ModelRegistry[Model Registry<br/>MLflow/Neptune] -.Embedding Model Versioning.-> Embedding

    style OCSF fill:#ADD8E6
    style Preprocessor fill:#FFFFE0
    style Embedding fill:#90EE90
    style VectorDB fill:#FFD700
    style Detector fill:#FFA500
    style AlertManager fill:#FF6347
    style Observability fill:#DDA0DD
    style ModelRegistry fill:#F5DEB3
```

---

## 1. Model Serving with FastAPI

### Near Real-Time Inference Pipeline

The FastAPI service implements the following pipeline for each incoming OCSF event:

1. **Preprocess**: Extract and encode features (categorical → embeddings, numerical → normalized)
2. **Generate Embedding**: Pass features through TabularResNet → 64-dimensional vector
3. **Similarity Search**: Query vector DB for k=20 nearest neighbor embeddings from historical baseline
4. **Anomaly Scoring**: Compute average distance to neighbors → higher distance = more anomalous
5. **Return Result**: Binary prediction (anomaly/normal) + confidence score

**Latency target**: <50ms per event (p99)

### Basic REST API

```{code-cell}
:tags: [skip-execution]

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict

app = FastAPI(title="OCSF Anomaly Detection API", version="1.0.0")

# Global model storage (loaded at startup)
MODEL = None
SCALER = None
ENCODERS = None
VECTOR_DB = None

class OCSFRecord(BaseModel):
    """OCSF record schema."""
    network_bytes_in: float
    duration: float
    user_id: str
    status_id: str
    entity_id: str
    # Add more fields as needed

class AnomalyResponse(BaseModel):
    """Anomaly detection response."""
    is_anomaly: bool
    anomaly_score: float
    confidence: float

@app.on_event("startup")
async def load_models():
    """Load models at startup."""
    global MODEL, SCALER, ENCODERS, VECTOR_DB

    # Load TabularResNet
    checkpoint = torch.load('models/ocsf_anomaly_detector.pt', map_location='cpu')
    MODEL = initialize_model(checkpoint['hyperparameters'])
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()

    SCALER = checkpoint['scaler']
    ENCODERS = checkpoint['encoders']

    # Initialize vector DB client (pseudo-interface)
    VECTOR_DB = init_vector_db_client(index_name="ocsf-embeddings")

    print("Models loaded successfully")

@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(record: OCSFRecord):
    """
    Predict if an OCSF record is anomalous.

    Args:
        record: OCSF record

    Returns:
        Anomaly prediction with score
    """
    try:
        # 1. Preprocess: Extract and encode features
        numerical, categorical = preprocess_record(record, SCALER, ENCODERS)

        # 2. Generate embedding: TabularResNet forward pass → 64-dim vector
        with torch.no_grad():
            embedding = MODEL(numerical, categorical, return_embedding=True)
            embedding_np = embedding.numpy()

        # 3. Similarity search: Query vector DB for k=20 nearest neighbors
        neighbors = VECTOR_DB.search(embedding_np, top_k=20)

        # 4. Anomaly scoring: Average distance to neighbors
        distances = [d for _, d in neighbors]
        score = float(np.mean(distances))
        threshold = float(np.percentile(distances, 95))
        prediction = score > threshold

        # 5. Return result
        is_anomaly = bool(prediction)
        confidence = abs(score)  # Higher = more confident

        return AnomalyResponse(
            is_anomaly=is_anomaly,
            anomaly_score=float(score),
            confidence=float(confidence)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[AnomalyResponse])
async def predict_batch(records: List[OCSFRecord]):
    """Batch prediction for multiple records."""
    results = []
    for record in records:
        result = await predict_anomaly(record)
        results.append(result)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "vector_db_connected": VECTOR_DB is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    # Return metrics in Prometheus format
    return {
        "predictions_total": 0,  # Increment in production
        "anomalies_detected": 0,
        "avg_inference_time_ms": 0.0
    }

def preprocess_record(record: OCSFRecord, scaler, encoders):
    """Preprocess a single OCSF record."""
    # Extract numerical features
    numerical = np.array([[
        record.network_bytes_in,
        record.duration
    ]])
    numerical = scaler.transform(numerical)
    numerical = torch.FloatTensor(numerical)

    # Encode categorical features
    categorical = []
    for field in ['user_id', 'status_id', 'entity_id']:
        value = getattr(record, field)
        if field in encoders:
            try:
                encoded = encoders[field].transform([value])[0]
            except ValueError:
                # Unknown category
                encoded = 0
            categorical.append(encoded)

    categorical = torch.LongTensor([categorical])

    return numerical, categorical

def initialize_model(hyperparams):
    """Initialize TabularResNet from hyperparameters."""
    # Import TabularResNet from Part 2
    from part2_tabular_resnet import TabularResNet

    return TabularResNet(
        num_numerical_features=hyperparams['num_numerical'],
        categorical_cardinalities=hyperparams['categorical_cardinalities'],
        d_model=hyperparams['d_model'],
        num_blocks=hyperparams['num_blocks'],
        dropout=0.1,
        num_classes=None
    )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
print("FastAPI application defined")
print("To run: uvicorn app:app --reload")
```

---

## 2. Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile for OCSF Anomaly Detection Service
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models/ ./models/
COPY utils/ ./utils/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### requirements.txt

```
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3
pydantic==2.5.0
joblib==1.3.2
prometheus-client==0.19.0
```

### docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    container_name: ocsf-anomaly-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models/ocsf_anomaly_detector.pt
      - VECTOR_DB_URL=http://vector-db:6333
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

---

## 3. Model Versioning & A/B Testing

### Model Registry with MLflow

```{code-cell}
import mlflow
import mlflow.pytorch

def register_model(model, scaler, encoders, metrics, experiment_name="ocsf-anomaly-detection"):
    """
    Register model with MLflow.

    Args:
        model: Trained TabularResNet
        scaler: Fitted scaler
        encoders: Categorical encoders
        metrics: Dict of evaluation metrics
        experiment_name: MLflow experiment name

    Returns:
        model_version
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log artifacts
        import joblib
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(encoders, "encoders.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("encoders.pkl")

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log parameters
        mlflow.log_param("d_model", model.d_model if hasattr(model, 'd_model') else 256)
        mlflow.log_param("num_blocks", len(model.blocks) if hasattr(model, 'blocks') else 6)

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_version = mlflow.register_model(model_uri, "OCSF-Anomaly-Detector")

        print(f"Model registered: version {model_version.version}")

    return model_version

# Example usage
"""
metrics = {
    'silhouette_score': 0.65,
    'f1_score': 0.82,
    'precision': 0.85,
    'recall': 0.79
}

model_version = register_model(model, scaler, encoders, metrics)
"""

print("MLflow model registration function defined")
```

### A/B Testing Framework

```{code-cell}
import random
from enum import Enum

class ModelVersion(Enum):
    """Model versions for A/B testing."""
    CHAMPION = "v1.0-champion"
    CHALLENGER = "v1.1-challenger"

class ABTestRouter:
    """
    Route traffic between champion and challenger models.
    """
    def __init__(self, champion_model, challenger_model, traffic_split=0.1):
        """
        Args:
            champion_model: Current production model
            challenger_model: New model to test
            traffic_split: Fraction of traffic to challenger (0.0-1.0)
        """
        self.champion = champion_model
        self.challenger = challenger_model
        self.traffic_split = traffic_split

        self.champion_requests = 0
        self.challenger_requests = 0

    def route(self, record):
        """
        Route request to champion or challenger.

        Args:
            record: Input record

        Returns:
            (prediction, model_version)
        """
        if random.random() < self.traffic_split:
            # Route to challenger
            self.challenger_requests += 1
            prediction = self.challenger.predict(record)
            version = ModelVersion.CHALLENGER
        else:
            # Route to champion
            self.champion_requests += 1
            prediction = self.champion.predict(record)
            version = ModelVersion.CHAMPION

        return prediction, version

    def get_stats(self):
        """Get routing statistics."""
        total = self.champion_requests + self.challenger_requests
        return {
            'champion_requests': self.champion_requests,
            'challenger_requests': self.challenger_requests,
            'total_requests': total,
            'champion_pct': self.champion_requests / total if total > 0 else 0,
            'challenger_pct': self.challenger_requests / total if total > 0 else 0
        }

# Example
print("A/B Testing Router defined")
print("Usage: router = ABTestRouter(champion, challenger, traffic_split=0.1)")
```

---

## 4. Real-Time vs Batch Inference

### Real-Time Inference (Streaming)

```{code-cell}
from kafka import KafkaConsumer, KafkaProducer
import json

class StreamingAnomalyDetector:
    """
    Real-time anomaly detection from Kafka streams.
    """
    def __init__(self, model, vector_db, kafka_brokers, input_topic, output_topic):
        """
        Args:
            model: TabularResNet model
            vector_db: Vector database client for k-NN retrieval
            kafka_brokers: List of Kafka broker addresses
            input_topic: Kafka topic for OCSF records
            output_topic: Kafka topic for alerts
        """
        self.model = model
        self.vector_db = vector_db

        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.output_topic = output_topic

    def process_stream(self):
        """
        Process incoming OCSF records in near real-time.
        """
        print("Streaming anomaly detection started...")

        for message in self.consumer:
            try:
                record = message.value

                # Preprocess
                numerical, categorical = preprocess_ocsf(record)

                # Generate embedding
                with torch.no_grad():
                    embedding = self.model(numerical, categorical, return_embedding=True)

                # Detect anomaly via vector DB retrieval
                neighbors = self.vector_db.search(embedding.numpy(), top_k=20)
                distances = [d for _, d in neighbors]
                score = float(np.mean(distances))
                threshold = float(np.percentile(distances, 95))
                is_anomaly = score > threshold

                if is_anomaly:
                    alert = {
                        'record_id': record.get('id'),
                        'timestamp': record.get('timestamp'),
                        'anomaly_score': float(score),
                        'details': record
                    }

                    # Publish alert
                    self.producer.send(self.output_topic, value=alert)
                    print(f"Anomaly detected: {alert['record_id']}")

            except Exception as e:
                print(f"Error processing record: {e}")
                continue

def preprocess_ocsf(record):
    """Preprocess OCSF JSON record."""
    # Implementation depends on your schema
    pass

print("StreamingAnomalyDetector class defined")
print("Usage: detector = StreamingAnomalyDetector(model, vector_db, brokers, 'ocsf-input', 'anomaly-alerts')")
```

### Batch Inference

```{code-cell}
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class BatchAnomalyDetector:
    """
    Batch anomaly detection for large datasets.
    """
    def __init__(self, model, detector, batch_size=1000, num_workers=4):
        """
        Args:
            model: TabularResNet model
            detector: Anomaly detector
            batch_size: Number of records per batch
            num_workers: Number of parallel workers
        """
        self.model = model
        self.detector = detector
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_batch(self, records_df):
        """
        Process a batch of records.

        Args:
            records_df: DataFrame with OCSF records

        Returns:
            DataFrame with anomaly predictions
        """
        # Preprocess all records
        numerical, categorical = preprocess_dataframe(records_df)

        # Generate embeddings (batch inference)
        with torch.no_grad():
            embeddings = self.model(numerical, categorical, return_embedding=True)
            embeddings_np = embeddings.numpy()

        # Detect anomalies
        predictions = self.detector.predict(embeddings_np)
        scores = self.detector.score_samples(embeddings_np)

        # Add results to dataframe
        records_df['is_anomaly'] = (predictions == -1)
        records_df['anomaly_score'] = scores

        return records_df

    def process_file(self, input_path, output_path):
        """
        Process entire file with parallel batching.

        Args:
            input_path: Path to input CSV/Parquet
            output_path: Path to output file
        """
        # Read file in chunks
        chunks = pd.read_csv(input_path, chunksize=self.batch_size)

        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_batch, chunk) for chunk in chunks]

            for future in futures:
                result = future.result()
                results.append(result)

        # Combine results
        final_df = pd.concat(results, ignore_index=True)

        # Save
        final_df.to_csv(output_path, index=False)
        print(f"Batch processing complete: {len(final_df)} records processed")
        print(f"Anomalies detected: {final_df['is_anomaly'].sum()}")

        return final_df

def preprocess_dataframe(df):
    """Preprocess DataFrame to tensors."""
    # Implementation depends on your schema
    pass

print("BatchAnomalyDetector class defined")
```

---

## 5. Performance Optimization

### Model Quantization

```{code-cell}
import torch.quantization

def quantize_model(model):
    """
    Quantize model for faster inference.

    Args:
        model: PyTorch model

    Returns:
        Quantized model (smaller, faster)
    """
    model.eval()

    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with sample data (use a small validation set)
    # calibration_data = ...
    # with torch.no_grad():
    #     for data in calibration_data:
    #         model(data)

    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)

    print("Model quantized successfully")
    print(f"Original size: {get_model_size(model):.2f} MB")
    print(f"Quantized size: {get_model_size(quantized_model):.2f} MB")

    return quantized_model

def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

print("Model quantization function defined")
```

### Caching Strategy

```{code-cell}
from functools import lru_cache
import hashlib

class EmbeddingCache:
    """
    Cache embeddings for frequently seen records.
    """
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, record):
        """Generate cache key from record."""
        # Hash relevant fields
        key_string = f"{record.user_id}_{record.status_id}_{record.entity_id}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, record):
        """Get cached embedding if exists."""
        key = self.get_key(record)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(self, record, embedding):
        """Cache embedding."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))

        key = self.get_key(record)
        self.cache[key] = embedding

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

print("EmbeddingCache class defined")
print("Use to cache embeddings for repeated records")
```

---

## Summary

In this part, you learned:

1. **REST API deployment** with FastAPI and Docker
2. **Model versioning** with MLflow
3. **A/B testing** framework for gradual rollout
4. **Real-time streaming** inference with Kafka
5. **Batch processing** for large-scale analysis
6. **Performance optimization** (quantization, caching)

**Key Deployment Patterns:**
- **Real-time**: Low-latency (<100ms), streaming data
- **Batch**: High-throughput, historical analysis
- **Hybrid**: Real-time alerts + daily batch reports

**Next**: In [Part 8](part8-production-monitoring), we'll monitor the deployed system for embedding drift, alert quality, and model performance degradation.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

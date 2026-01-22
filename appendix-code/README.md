# OCSF Training Data Generator

Generate realistic observability data in OCSF format for **self-supervised** anomaly detection training.

## Architecture

All telemetry flows through **OpenTelemetry Collector**:
- **Logs**: Services -> OTLP -> OTel Collector -> logs.jsonl
- **Traces**: Services -> OTLP -> OTel Collector -> traces.jsonl
- **Metrics**: OTel Collector scrapes /metrics -> metrics.jsonl

## What This Generates

- Realistic observability data from a multi-service application
- Normal traffic + anomaly scenarios (unlabeled)
- **Logs**, **metrics**, and **traces** in OCSF format
- No labels required - use with self-supervised learning

## Quick Start

```bash
# 1. Create required directories (with write permissions for containers)
# Note: If you previously ran docker compose, the directory may be root-owned
sudo rm -rf ./logs/otel 2>/dev/null || true
mkdir -p ./logs/otel ./data
chmod 777 ./logs/otel

# 2. Build services (ensures latest code)
docker compose build

# 3. Start all services
docker compose up -d

# 4. Let the load generator run (5-10 min for demo, 2 hours for full dataset)
# Traffic is automatically generated

# 5. Convert all OpenTelemetry data to OCSF format:
python scripts/convert_otel_to_ocsf.py

# 6. (Optional) Generate small labeled subset for evaluation
python scripts/label_subset_for_evaluation.py
```

## Prerequisites

- Docker 20.10+ with Compose plugin (uses `docker compose`, not `docker-compose`)
- Python 3.8+ with: `pip install pandas pyarrow`

## Output Datasets

After running the conversion script:

- `data/ocsf_logs.parquet` - Application logs in OCSF format (unlabeled)
- `data/ocsf_traces.parquet` - Distributed traces in OCSF format (unlabeled)
- `data/ocsf_metrics.parquet` - System metrics in OCSF format (unlabeled)
- `data/ocsf_eval_subset.parquet` - Small labeled subset for evaluation (optional)

## Stack Components

| Service | Role | Telemetry |
|---------|------|-----------|
| **web-api** | Flask service with observability | Logs + traces via OTLP, metrics via /metrics |
| **auth-service** | Node.js authentication service | Logs via stdout |
| **payment-worker** | Background job processor | Logs via stdout |
| **postgres** | Database | - |
| **redis** | Cache | - |
| **otel-collector** | Unified telemetry hub | Exports all signals to JSONL |
| **load-generator** | Traffic patterns with anomalies | - |

## Conversion Script

| Script | Input | Output |
|--------|-------|--------|
| `convert_otel_to_ocsf.py` | `logs/otel/*.jsonl` | `data/ocsf_*.parquet` |

Convert specific signals:
```bash
python scripts/convert_otel_to_ocsf.py --signal logs
python scripts/convert_otel_to_ocsf.py --signal traces
python scripts/convert_otel_to_ocsf.py --signal metrics
```

## Use Cases

- Self-supervised training with TabularResNet (Part 4)
- Unsupervised anomaly detection (Part 6: LOF, Isolation Forest, k-NN)
- Testing observability systems
- Learning OCSF schema

## Important: No Labels Needed

This generator creates **unlabeled** data that works with:
- **Self-supervised learning** (Part 4: contrastive learning, masked prediction)
- **Unsupervised anomaly detection** (Part 6: LOF, Isolation Forest, k-NN)

Labels are only needed for evaluation/comparison (Part 6, Section 7), not training.

## Stopping Services

```bash
docker compose down      # Stop services
docker compose down -v   # Stop and remove volumes
```

## License

MIT - Free to use for commercial and non-commercial purposes

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

# Part 3: Feature Engineering for OCSF Data

Learn how to transform raw OCSF security logs into the feature arrays that TabularResNet expects.

**Note**: If you don't have OCSF observability data to work with, see [Appendix: Generating Training Data](appendix-generating-training-data) to learn how to spin up a Docker Compose stack that generates realistic logs, metrics, and traces with labeled anomalies.

**The challenge**: You have raw OCSF JSON events with nested fields, optional values, and 300+ possible attributes. TabularResNet needs clean numerical and categorical arrays. This part bridges that gap.

**What you'll learn**:
1. Loading and parsing OCSF JSON structure
2. Extracting and flattening nested fields
3. Engineering temporal, aggregation, and derived features
4. Handling missing values and high cardinality
5. Building an end-to-end feature pipeline

---

## Understanding OCSF Structure

The [Open Cybersecurity Schema Framework (OCSF)](https://schema.ocsf.io/) provides a standardized schema for security events. Each event has:

- **class_name**: Event type (Authentication, Network Activity, File Activity, etc.)
- **Core fields**: severity_id, time, activity_id, status_id
- **Nested objects**: actor, src_endpoint, dst_endpoint, http_request, etc.
- **Optional fields**: Many fields are present only for specific event types

**Example OCSF Authentication Event**:

```json
{
  "class_name": "Authentication",
  "class_uid": 3002,
  "severity_id": 1,
  "activity_id": 1,
  "time": 1672531200000,
  "status_id": 1,
  "actor": {
    "user": {
      "name": "john.doe",
      "uid": "12345",
      "email": "john.doe@company.com"
    },
    "session": {
      "uid": "sess-98765",
      "created_time": 1672531195000
    }
  },
  "src_endpoint": {
    "ip": "192.168.1.100",
    "port": 54321,
    "location": {
      "city": "San Francisco",
      "country": "US"
    }
  },
  "http_request": {
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "url": {
      "hostname": "api.company.com",
      "path": "/auth/login"
    }
  },
  "auth_protocol": "OAuth 2.0",
  "logon_type": "Interactive"
}
```

**Key observations**:
- Deeply nested structure (actor.user.name is 3 levels deep)
- Mix of strings, numbers, timestamps
- Some fields always present, others optional
- High-cardinality fields (user.uid, session.uid)

---

## Loading OCSF Data

**What we're doing**: Reading OCSF events from storage into Python dictionaries for processing.

**Why**: OCSF data typically arrives in newline-delimited JSON format (`.jsonl`) where each line is a complete event. This format is:
- Space-efficient for large datasets
- Streamable (process one event at a time without loading everything into memory)
- Standard format for security log exports

**Pitfalls**:
- **Memory**: Loading all events at once can exhaust memory. For large datasets (millions of events), use generators or process in batches
- **Malformed JSON**: Production logs often contain corrupted lines. Always wrap `json.loads()` in try/except
- **Encoding issues**: Security logs may contain non-UTF-8 characters. Use `encoding='utf-8', errors='replace'` when opening files

### From JSON Files

```{code-cell}
:tags: [skip-execution]

import json
import pandas as pd

def load_ocsf_from_file(filepath):
    """Load OCSF events from newline-delimited JSON file."""
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # Log malformed lines instead of crashing
                    print(f"Skipping malformed JSON: {e}")
                    continue
    return events

# Example usage
events = load_ocsf_from_file('ocsf_events.jsonl')
print(f"Loaded {len(events)} OCSF events")
```

**Further processing**: For production systems with large datasets, use a generator pattern:

```python
def load_ocsf_generator(filepath):
    """Memory-efficient generator for OCSF events."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

# Process one at a time without loading all into memory
for event in load_ocsf_generator('ocsf_events.jsonl'):
    features = extract_features(event)
    # ... process features
```

### From Streaming Sources

**What we're doing**: Consuming OCSF events from a Kafka stream for near real-time processing.

**Why**: Production security systems generate millions of events per day. Kafka provides:
- **Buffering**: Handles burst traffic without data loss
- **Scalability**: Multiple consumers can process events in parallel
- **Replay**: Can reprocess historical events for retraining models

**How it works**:
1. Consumer polls Kafka topic for new messages
2. Each message contains one OCSF event (JSON string)
3. Decode bytes to UTF-8 and parse JSON
4. Yield event for feature extraction

```{code-cell}
:tags: [skip-execution]

# Kafka example (using confluent-kafka)
from confluent_kafka import Consumer

def consume_ocsf_from_kafka(topic, bootstrap_servers):
    """
    Consume OCSF events from Kafka topic.

    Args:
        topic: Kafka topic name
        bootstrap_servers: Kafka broker addresses

    Yields:
        Parsed OCSF event dictionaries
    """
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': 'ocsf-feature-engineering',
        'auto.offset.reset': 'earliest'  # Start from beginning if no offset
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])

    try:
        while True:
            msg = consumer.poll(1.0)  # 1 second timeout
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Parse OCSF JSON
            event = json.loads(msg.value().decode('utf-8'))
            yield event

    finally:
        consumer.close()

# Usage
# for event in consume_ocsf_from_kafka('ocsf-events', 'localhost:9092'):
#     features = extract_features(event)
```

**Pitfalls**:
- **Backpressure**: If feature extraction is slow, Kafka consumer lag will grow. Monitor consumer lag metrics and scale consumers horizontally if needed
- **Deserialization errors**: Kafka messages may contain invalid JSON. Always wrap parsing in try/except
- **Offset management**: If processing crashes, `auto.offset.reset` determines whether to replay or skip events. For training, use 'earliest' to reprocess all data. For production inference, use 'latest' to process only new events

---

## Extracting Raw Features

### Flattening Nested Fields

**What we're doing**: Converting nested JSON dictionaries into flat key-value pairs.

**Why**: TabularResNet expects flat feature vectors, not nested objects. OCSF uses deep nesting (e.g., `actor.user.name` is 3 levels deep), so we need to flatten before feature extraction.

**How it works**:
- Recursively traverse the dictionary tree
- Concatenate parent keys with child keys using underscores
- Example: `{"actor": {"user": {"name": "john"}}}` → `{"actor_user_name": "john"}`
- Arrays: Take first element (most arrays in OCSF have single values) and add a `_count` field for length

```{code-cell}
import pandas as pd

def flatten_ocsf_event(event, prefix='', sep='_'):
    """
    Recursively flatten nested OCSF event structure.

    Args:
        event: OCSF event dictionary
        prefix: Current key prefix for recursion
        sep: Separator for nested keys

    Returns:
        Flat dictionary with dot-notation keys

    Example:
        {'actor': {'user': {'name': 'john'}}}
        → {'actor_user_name': 'john'}
    """
    flat = {}

    for key, value in event.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat.update(flatten_ocsf_event(value, new_key, sep))
        elif isinstance(value, list):
            # For lists, take first element or length
            if len(value) > 0:
                if isinstance(value[0], dict):
                    flat.update(flatten_ocsf_event(value[0], new_key, sep))
                else:
                    flat[new_key] = value[0]
            flat[f"{new_key}_count"] = len(value)
        else:
            flat[new_key] = value

    return flat

# Example
event = {
    "class_name": "Authentication",
    "severity_id": 1,
    "actor": {
        "user": {"name": "john.doe", "uid": "12345"}
    },
    "src_endpoint": {
        "ip": "192.168.1.100"
    }
}

flat_event = flatten_ocsf_event(event)
print("Flattened event:")
for key, value in flat_event.items():
    print(f"  {key}: {value}")
```

**Pitfalls**:
- **Name collisions**: If keys at different levels have same name, they'll collide. Example: `{"user": {"id": 1}, "id": 2}` → both become `id`. Use more specific key names or include full path
- **Array handling**: Taking only the first element loses information if arrays have multiple values. For security logs with multiple IPs or ports, consider extracting all values or computing summary statistics (min/max/count)
- **Explosion of features**: Deep nesting creates many features. A 5-level nested object can produce 50+ flattened keys. Filter to only useful fields after flattening

**Further processing needed**: After flattening, you'll have 100-300 fields. Next step is feature selection to choose the 20-50 most informative ones.

### Selecting Core Features

**What we're doing**: Choosing the most informative subset of OCSF fields for anomaly detection.

**Why**: Not all 300+ OCSF fields are useful. Many are:
- Always null for your data source
- Redundant (e.g., `time` and `time_dt` contain same information)
- Too high cardinality (e.g., unique message text)
- Not predictive of anomalies

Starting with 20-50 core features keeps the model focused and reduces overfitting.

**How to choose**:
1. **Domain knowledge**: Security experts know which fields matter (user_id, IP addresses, status codes)
2. **Data exploration**: Check which fields have non-null values >90% of the time
3. **Tree-based importance**: Train Random Forest on sample data and rank features by importance score (see [Tree-Based Feature Importance](/ai-eng/ml-algorithms/tree_based_feature_importance.md) for detailed guide)

**LLM-assisted feature selection**: If you lack security domain expertise, use an LLM to recommend features based on your use case:

<details>
<summary><strong>Click to see LLM prompt template</strong></summary>

```
I'm building an anomaly detection system for OCSF (Open Cybersecurity Schema Framework) security logs.
I need to select 20-50 most informative features from the 300+ available OCSF fields.

My specific use case:
- [Describe your use case: e.g., "Detect compromised user accounts", "Identify data exfiltration",
  "Monitor API access patterns", "Detect brute force attacks"]

My data sources:
- [List your log sources: e.g., "AWS CloudTrail logs", "Okta authentication events",
  "Kubernetes audit logs", "Web application access logs"]

Based on this use case, please recommend:
1. The top 20-30 OCSF fields I should extract as features
2. For each field, explain why it's important for detecting my target anomalies
3. Categorize them as categorical vs numerical features
4. Highlight any derived features I should engineer (e.g., ratios, aggregations)
5. Suggest appropriate time windows for aggregation features (e.g., 1 hour, 24 hours)

Example OCSF schema reference: https://schema.ocsf.io/
```

**Example response** (for detecting compromised accounts):
```
Recommended features:
1. actor.user.uid (categorical) - Track which user account
2. status_id (categorical) - Success/failure patterns
3. src_endpoint.ip (categorical, hashed) - Login locations
4. time (numerical) - Extract hour_of_day, day_of_week
5. Failed_login_count_1h (derived, numerical) - Brute force indicator
6. unique_ip_count_24h (derived, numerical) - Account sharing indicator
7. geo_distance_from_prev (derived, numerical) - Impossible travel detection
...
```
</details>

```{code-cell}
def extract_core_features(event):
    """
    Extract core features from OCSF event.

    Returns dictionary with categorical and numerical features.
    """
    flat = flatten_ocsf_event(event)

    # Categorical features (strings/IDs that represent categories)
    categorical = {
        'class_name': flat.get('class_name'),
        'severity_id': flat.get('severity_id'),
        'activity_id': flat.get('activity_id'),
        'status_id': flat.get('status_id'),
        'user_name': flat.get('actor_user_name'),
        'src_ip': flat.get('src_endpoint_ip'),
        'dst_ip': flat.get('dst_endpoint_ip'),
        'auth_protocol': flat.get('auth_protocol'),
        'http_method': flat.get('http_request_http_method'),
    }

    # Numerical features (continuous values)
    numerical = {
        'time': flat.get('time', 0),
        'src_port': flat.get('src_endpoint_port', 0),
        'dst_port': flat.get('dst_endpoint_port', 0),
        'bytes_in': flat.get('traffic_bytes_in', 0),
        'bytes_out': flat.get('traffic_bytes_out', 0),
        'duration': flat.get('duration', 0),
    }

    return categorical, numerical

# Example
cat_features, num_features = extract_core_features(event)
print("\nCategorical features:", cat_features)
print("Numerical features:", num_features)
```

---

## Engineering Temporal Features

**What we're doing**: Extracting time-based patterns from Unix timestamps.

**Why**: Temporal patterns are critical for anomaly detection because:
- **Normal behavior varies by time**: Logins at 3 AM are suspicious, but normal at 9 AM
- **Attack patterns have timing**: Brute force attacks happen rapidly; data exfiltration often happens off-hours
- **Cyclical patterns matter**: Monday mornings have different traffic than Sunday nights

**Types of temporal features**:
1. **Categorical**: hour_of_day (0-23), day_of_week (0-6), is_weekend, is_business_hours
2. **Cyclical (sin/cos)**: Preserves circular nature of time (23:00 is close to 00:00)
3. **Aggregations**: Time since last event, events per hour (covered in next section)

```{code-cell}
from datetime import datetime
import numpy as np

def extract_temporal_features(timestamp_ms):
    """
    Extract temporal features from Unix timestamp (milliseconds).

    Args:
        timestamp_ms: Unix timestamp in milliseconds

    Returns:
        Dictionary of temporal features (both categorical and numerical)
    """
    if timestamp_ms is None or timestamp_ms == 0:
        return {
            'hour_of_day': 0,
            'day_of_week': 0,
            'is_weekend': 0,
            'is_business_hours': 0,
            'hour_sin': 0.0,
            'hour_cos': 1.0,
            'day_sin': 0.0,
            'day_cos': 1.0,
        }

    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)

    # Basic temporal features
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday

    # Derived boolean features
    is_weekend = int(day_of_week >= 5)  # Saturday or Sunday
    is_business_hours = int(9 <= hour < 17)  # 9 AM to 5 PM

    # Cyclical encoding (hour and day wrap around)
    # Sin/cos encoding preserves circular nature (23:00 is close to 00:00)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    return {
        'hour_of_day': hour,  # Categorical: 0-23
        'day_of_week': day_of_week,  # Categorical: 0-6
        'is_weekend': is_weekend,  # Binary
        'is_business_hours': is_business_hours,  # Binary
        'hour_sin': hour_sin,  # Numerical cyclical
        'hour_cos': hour_cos,  # Numerical cyclical
        'day_sin': day_sin,  # Numerical cyclical
        'day_cos': day_cos,  # Numerical cyclical
    }

# Example
timestamp = 1672531200000  # 2023-01-01 00:00:00 UTC
temporal = extract_temporal_features(timestamp)
print("\nTemporal features:")
for key, value in temporal.items():
    print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
```

**Why cyclical encoding?**
- **Problem**: Hour 23 and hour 0 are only 1 hour apart, but numerically 23 units apart
- **Solution**: Encode as (sin, cos) pair on unit circle
- **Result**: Similar hours have similar encodings (23:00 and 00:00 are close in (sin, cos) space)

**How it works**:
- Map hour to angle: `angle = 2π × hour / 24`
- Convert to (sin, cos): `(sin(angle), cos(angle))`
- Hours near each other → similar (sin, cos) values
- Example: hour=0 → (0, 1), hour=23 → (≈-0.26, ≈0.97) are close in Euclidean space

**Further processing**: These temporal features will be:
- **Categorical** (hour_of_day, day_of_week, is_weekend, is_business_hours): Fed to categorical embeddings in TabularResNet
- **Numerical** (hour_sin, hour_cos, day_sin, day_cos): Normalized with StandardScaler before feeding to TabularResNet

**Pitfalls**:
- **Timezone issues**: OCSF timestamps are typically UTC. If your normal behavior patterns are timezone-specific (e.g., logins spike at 9 AM local time), convert to local timezone before extracting hour
- **Categorical vs cyclical**: Don't use both raw hour (0-23) AND (hour_sin, hour_cos) as features. They encode the same information - use cyclical for better gradient flow

---

## Engineering Derived Features

**What we're doing**: Creating new features by combining or transforming existing fields.

**Why**: Raw OCSF fields often need transformation to be useful:
- **Rates matter more than totals**: `bytes_per_second` is more informative than raw `bytes` value
- **Ratios reveal patterns**: `upload_ratio` (upload/total) can detect data exfiltration
- **Domain features**: Extracting TLD from URLs can reveal phishing (unusual TLDs like `.tk`, `.ru`)

**Examples**:
- Network transfer rates: `bytes_per_second = total_bytes / duration`
- Upload/download ratio: `upload_ratio = bytes_out / (bytes_in + bytes_out)`
- Domain features: Extract TLD, domain length from hostnames
- Boolean indicators: `is_default_port`, `has_user_agent`

```{code-cell}
def extract_derived_features(event):
    """
    Create derived features from raw OCSF fields.

    Examples: rates, ratios, domain extraction, etc.
    """
    flat = flatten_ocsf_event(event)

    features = {}

    # Network transfer rates
    duration = flat.get('duration', 0)
    bytes_in = flat.get('traffic_bytes_in', 0)
    bytes_out = flat.get('traffic_bytes_out', 0)

    if duration > 0:
        features['bytes_per_second'] = (bytes_in + bytes_out) / (duration / 1000.0)
    else:
        features['bytes_per_second'] = 0.0

    # Total bytes
    features['total_bytes'] = bytes_in + bytes_out

    # Byte ratio (upload vs download)
    if bytes_in + bytes_out > 0:
        features['upload_ratio'] = bytes_out / (bytes_in + bytes_out)
    else:
        features['upload_ratio'] = 0.5

    # Domain extraction from URL
    hostname = flat.get('http_request_url_hostname', '')
    if hostname:
        # Extract top-level domain
        parts = hostname.split('.')
        features['tld'] = parts[-1] if len(parts) > 0 else 'unknown'
        features['domain_length'] = len(hostname)
    else:
        features['tld'] = 'unknown'
        features['domain_length'] = 0

    # User agent analysis
    user_agent = flat.get('http_request_user_agent', '')
    features['has_user_agent'] = int(len(user_agent) > 0)
    features['user_agent_length'] = len(user_agent)

    # Check for suspicious patterns
    features['is_default_port'] = int(flat.get('dst_endpoint_port') in [80, 443, 22, 21])

    return features

# Example
event_with_traffic = {
    "duration": 5000,  # 5 seconds
    "traffic": {"bytes_in": 1024000, "bytes_out": 512000},
    "http_request": {
        "url": {"hostname": "api.company.com"},
        "user_agent": "Mozilla/5.0..."
    },
    "dst_endpoint": {"port": 443}
}

derived = extract_derived_features(event_with_traffic)
print("\nDerived features:")
for key, value in derived.items():
    print(f"  {key}: {value}")
```

---

## Engineering Aggregation Features

**What we're doing**: Computing statistics over recent events (rolling windows) to capture behavioral patterns.

**Why this is critical for anomaly detection**:
- **Single events lack context**: One failed login is normal; 50 in 10 minutes is a brute force attack
- **Behavioral baselines**: How many events does this user normally generate per hour?
- **Velocity features**: Rapid changes in behavior (e.g., sudden spike in data transfer) are anomalies

**How it works**:
1. Maintain a sliding window of recent events per user (e.g., last 60 minutes)
2. For each new event, compute statistics over the user's recent events
3. Remove events outside the time window to keep memory bounded
4. Features: event counts, failure rates, unique IPs, average bytes, time since last event

**Memory management**: Use `deque` with `maxlen` or timestamp-based pruning to prevent unbounded memory growth.

```{code-cell}
:tags: [skip-execution]

from collections import defaultdict, deque
from datetime import datetime, timedelta

class FeatureAggregator:
    """
    Maintain rolling window aggregations for OCSF events.

    Tracks per-user statistics over time windows.
    """

    def __init__(self, window_minutes=60):
        self.window_minutes = window_minutes
        self.window_seconds = window_minutes * 60

        # Store recent events per user
        self.user_events = defaultdict(lambda: deque())

    def add_event(self, event):
        """Add event and compute aggregations."""
        flat = flatten_ocsf_event(event)
        user = flat.get('actor_user_name', 'unknown')
        timestamp = flat.get('time', 0) / 1000.0  # Convert to seconds

        # Add to user's event history
        self.user_events[user].append({
            'timestamp': timestamp,
            'event': flat
        })

        # Remove events outside window
        cutoff = timestamp - self.window_seconds
        while (self.user_events[user] and
               self.user_events[user][0]['timestamp'] < cutoff):
            self.user_events[user].popleft()

        # Compute aggregations
        return self._compute_aggregations(user, flat)

    def _compute_aggregations(self, user, current_event):
        """Compute aggregations over user's recent events."""
        recent = self.user_events[user]

        if len(recent) == 0:
            return {
                'event_count_1h': 0,
                'failed_count_1h': 0,
                'unique_ips_1h': 0,
                'avg_bytes_1h': 0.0,
                'time_since_last_event': 0.0,
            }

        events = [e['event'] for e in recent]

        # Count events
        event_count = len(events)

        # Count failures
        failed_count = sum(1 for e in events if e.get('status_id') != 1)

        # Unique source IPs
        unique_ips = len(set(e.get('src_endpoint_ip') for e in events
                            if e.get('src_endpoint_ip')))

        # Average bytes
        total_bytes = sum(e.get('traffic_bytes_in', 0) + e.get('traffic_bytes_out', 0)
                         for e in events)
        avg_bytes = total_bytes / event_count if event_count > 0 else 0.0

        # Time since last event
        if len(recent) >= 2:
            time_since_last = (recent[-1]['timestamp'] -
                             recent[-2]['timestamp'])
        else:
            time_since_last = 0.0

        return {
            'event_count_1h': event_count,
            'failed_count_1h': failed_count,
            'unique_ips_1h': unique_ips,
            'avg_bytes_1h': avg_bytes,
            'time_since_last_event': time_since_last,
        }

# Usage example
aggregator = FeatureAggregator(window_minutes=60)

# Process events
for event in events:
    agg_features = aggregator.add_event(event)
    # Combine with other features
    all_features = {**extract_core_features(event), **agg_features}
```

**Aggregation features are powerful for anomaly detection**:
- Sudden spike in failed logins → Brute force attack
- Many unique IPs from one user → Compromised account
- Unusual bytes transferred → Data exfiltration

**Pitfalls**:
- **Cold start problem**: First event for a user has no history. Aggregations return 0, which may be flagged as anomalous. Solution: Have a "warm-up period" or special handling for new users
- **Memory growth**: Tracking millions of users indefinitely exhausts memory. Solution: Periodically purge inactive users (no events in N hours), or use external state store (Redis) for production
- **Out-of-order events**: If events arrive out of timestamp order (common with distributed systems), aggregations may be incorrect. Solution: Buffer events and sort by timestamp before processing, or use a stateless approach (query from database)
- **Window size choice**: Too small (5 minutes) → noisy, too large (24 hours) → misses short attacks. Start with 60 minutes for most security use cases

**Further processing**: For production systems processing millions of events:
- **Use external state**: Store user aggregations in Redis with TTL (time-to-live) instead of in-memory dictionaries
- **Batch processing**: For training data, precompute aggregations in Spark/Dask rather than streaming aggregator
- **Multiple windows**: Compute 1h, 4h, and 24h aggregations to capture different attack timescales

---

## Handling Missing Values

OCSF events have sparse, optional fields. Strategy:

```{code-cell}
def handle_missing_values(features_dict, categorical_cols, numerical_cols):
    """
    Handle missing values in feature dictionary.

    Args:
        features_dict: Dictionary of extracted features
        categorical_cols: List of categorical feature names
        numerical_cols: List of numerical feature names

    Returns:
        Dictionary with missing values handled
    """
    processed = {}

    # Categorical: use special "MISSING" category
    for col in categorical_cols:
        value = features_dict.get(col)
        if value is None or value == '' or pd.isna(value):
            processed[col] = 'MISSING'
        else:
            processed[col] = str(value)

    # Numerical: use 0 or add binary indicator
    for col in numerical_cols:
        value = features_dict.get(col)
        if value is None or pd.isna(value):
            processed[col] = 0.0
            processed[f'{col}_is_missing'] = 1  # Binary indicator
        else:
            processed[col] = float(value)
            processed[f'{col}_is_missing'] = 0

    return processed

# Example
incomplete_features = {
    'user_name': 'john.doe',
    'src_ip': None,  # Missing
    'bytes_in': 1024,
    'duration': None,  # Missing
}

cat_cols = ['user_name', 'src_ip']
num_cols = ['bytes_in', 'duration']

processed = handle_missing_values(incomplete_features, cat_cols, num_cols)
print("\nProcessed features with missing value handling:")
for key, value in processed.items():
    print(f"  {key}: {value}")
```

---

## Handling High Cardinality

**What we're doing**: Reducing unbounded categorical features (millions of unique values) to fixed-size representations.

**Why**: Some OCSF fields have extreme cardinality:
- **IP addresses**: Millions of unique client IPs
- **User IDs**: Can be millions in large organizations
- **Session IDs**: Unique per login session
- **URLs/Paths**: Unbounded unique strings

TabularResNet's categorical embeddings need fixed vocabulary sizes. We can't create an embedding table with millions of rows.

**Two techniques**:
1. **Hashing trick**: Map unlimited values to fixed buckets (e.g., 1000)
2. **Subnet encoding**: For IPs, group by subnet (192.168.1.x) instead of full address

### Hashing Trick

**How it works**:
- Apply hash function to value (e.g., `hash("192.168.1.100")`)
- Modulo by number of buckets: `hash(value) % 1000` → bucket ID 0-999
- Different values usually map to different buckets (collisions are rare with good hash functions)
- **Tradeoff**: Some collisions are acceptable - model learns that bucket 456 represents "this group of similar IPs"

```{code-cell}
def hash_categorical_feature(value, num_buckets=1000):
    """
    Hash high-cardinality categorical to fixed number of buckets.

    Args:
        value: Original categorical value (string)
        num_buckets: Number of hash buckets

    Returns:
        Integer in range [0, num_buckets-1]
    """
    if value is None or value == '':
        return 0

    # Use Python's built-in hash
    return hash(str(value)) % num_buckets

# Example: Hash IP addresses to 1000 buckets
ip = "192.168.1.100"
hashed_ip = hash_categorical_feature(ip, num_buckets=1000)
print(f"\nOriginal IP: {ip}")
print(f"Hashed to bucket: {hashed_ip}")

# Different IPs map to different buckets (usually)
ips = ["192.168.1.100", "10.0.0.5", "172.16.0.1"]
for ip in ips:
    print(f"{ip} → bucket {hash_categorical_feature(ip, 1000)}")
```

**Pitfalls**:
- **Collision rate**: With 1M unique IPs and 1000 buckets, expect ~1000 collisions (birthday paradox). Increase buckets (10K-100K) if cardinality is very high
- **No semantic meaning**: Hashing loses all structure. IPs 192.168.1.100 and 192.168.1.101 (same subnet) map to random different buckets. Use subnet encoding if network structure matters
- **Inference mismatch**: Use same num_buckets in training and inference, otherwise hashed values won't align

### IP Subnet Encoding

**How it works**:
- Keep first 3 octets (subnet): `192.168.1.x`
- Maps all IPs in same /24 subnet to same category
- Reduces cardinality from millions to thousands
- **Advantage over hashing**: Preserves network structure - IPs from same subnet have same encoding

```{code-cell}
def encode_ip_subnet(ip_address):
    """
    Encode IP as subnet (keep first 3 octets, hash last).

    Preserves network structure while reducing cardinality.
    """
    if not ip_address:
        return 'MISSING'

    parts = ip_address.split('.')
    if len(parts) != 4:
        return 'INVALID'

    # Keep first 3 octets (subnet), anonymize last octet
    subnet = '.'.join(parts[:3])
    return f"{subnet}.x"

# Example
ips = ["192.168.1.100", "192.168.1.101", "10.0.0.5"]
for ip in ips:
    print(f"{ip} → {encode_ip_subnet(ip)}")
```

---

## End-to-End Feature Pipeline

Putting it all together:

```{code-cell}
:tags: [skip-execution]

class OCSFFeatureExtractor:
    """
    End-to-end feature extraction pipeline for OCSF events.
    """

    def __init__(self, window_minutes=60):
        self.aggregator = FeatureAggregator(window_minutes)

        # Define feature schema
        self.categorical_features = [
            'class_name', 'severity_id', 'activity_id', 'status_id',
            'user_name', 'src_ip_subnet', 'auth_protocol',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]

        self.numerical_features = [
            'src_port', 'dst_port', 'bytes_per_second', 'total_bytes',
            'upload_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'event_count_1h', 'failed_count_1h', 'unique_ips_1h',
            'avg_bytes_1h', 'time_since_last_event'
        ]

    def extract(self, event):
        """
        Extract all features from OCSF event.

        Returns:
            Dictionary with categorical and numerical features
        """
        # Core features
        cat_core, num_core = extract_core_features(event)

        # Temporal features
        timestamp = event.get('time', 0)
        temporal = extract_temporal_features(timestamp)

        # Derived features
        derived = extract_derived_features(event)

        # Aggregation features
        agg = self.aggregator.add_event(event)

        # Combine all features
        all_features = {**cat_core, **num_core, **temporal, **derived, **agg}

        # IP subnet encoding
        all_features['src_ip_subnet'] = encode_ip_subnet(
            all_features.get('src_ip'))

        # Handle missing values
        processed = handle_missing_values(
            all_features,
            self.categorical_features,
            self.numerical_features
        )

        return processed

    def batch_extract(self, events):
        """
        Extract features from multiple events.

        Returns:
            pandas DataFrame with one row per event
        """
        feature_dicts = [self.extract(event) for event in events]
        return pd.DataFrame(feature_dicts)

# Usage
extractor = OCSFFeatureExtractor(window_minutes=60)

# Process single event
features = extractor.extract(event)

# Process batch
# df = extractor.batch_extract(events)
# print(df.head())
```

---

## Preparing for TabularResNet

**What we're doing**: Converting DataFrame of mixed-type features into the numerical/categorical arrays TabularResNet expects.

**Why this is critical**:
- **TabularResNet needs specific input format**: Two separate arrays - one for numerical features (floats), one for categorical features (integers)
- **Normalization matters**: Neural networks train poorly with unnormalized inputs. StandardScaler ensures all numerical features have mean=0, std=1
- **Categorical encoding**: Convert string categories ("success", "failure") to integer indices (0, 1, 2, ...)
- **Save encoders/scalers**: Must use the SAME encoding at inference time, or model sees unknown values

**Steps**:
1. **Encode categoricals**: LabelEncoder maps strings → integers. Save encoder for inference
2. **Normalize numericals**: StandardScaler makes mean=0, std=1. Save scaler for inference
3. **Track cardinalities**: TabularResNet needs to know vocabulary size for each categorical (to create embedding tables)

```{code-cell}
:tags: [skip-execution]

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def prepare_for_tabular_resnet(df, categorical_cols, numerical_cols):
    """
    Prepare OCSF features for TabularResNet.

    Args:
        df: DataFrame with extracted features
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names

    Returns:
        numerical_array: (N, num_numerical) normalized array
        categorical_array: (N, num_categorical) integer-encoded array
        encoders: Dictionary of LabelEncoders for each categorical
        scaler: StandardScaler for numerical features
        categorical_cardinalities: List of unique values per categorical
    """
    # Encode categorical features
    encoders = {}
    categorical_data = []

    for col in categorical_cols:
        encoder = LabelEncoder()
        # Add 'UNKNOWN' to handle new values at inference time
        unique_vals = list(df[col].unique()) + ['UNKNOWN']
        encoder.fit(unique_vals)
        encoded = encoder.transform(df[col])
        categorical_data.append(encoded)
        encoders[col] = encoder

    categorical_array = np.column_stack(categorical_data)
    categorical_cardinalities = [len(enc.classes_) for enc in encoders.values()]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_array = scaler.fit_transform(df[numerical_cols])

    return (numerical_array, categorical_array,
            encoders, scaler, categorical_cardinalities)

# Usage
# numerical, categorical, encoders, scaler, cardinalities = \
#     prepare_for_tabular_resnet(df, categorical_cols, numerical_cols)

# Now ready for TabularResNet from Part 2!
# model = TabularResNet(
#     num_numerical_features=numerical.shape[1],
#     categorical_cardinalities=cardinalities,
#     d_model=256,
#     num_blocks=6
# )
```

**Pitfalls**:
- **New categories at inference**: If inference data contains categorical value "error_code=500" never seen in training, LabelEncoder will crash. Solution: Add 'UNKNOWN' to training vocabulary (as done above with `unique_vals + ['UNKNOWN']`) and map unseen values to it
- **Normalization leakage**: If you fit StandardScaler on full dataset (train + validation), validation scores are overly optimistic. Solution: Fit scaler only on training data, transform both train and validation with it
- **Feature order matters**: Categorical/numerical arrays must have same column order at inference as training. Solution: Save feature column names alongside encoders/scalers
- **Scaling before splitting**: NEVER scale data before train/test split. Always split first, then fit scaler on train only

**Further processing needed**:
- **Save artifacts**: Pickle encoders, scaler, feature names, cardinalities for inference
  ```python
  import pickle
  with open('feature_artifacts.pkl', 'wb') as f:
      pickle.dump({
          'encoders': encoders,
          'scaler': scaler,
          'categorical_cols': categorical_cols,
          'numerical_cols': numerical_cols,
          'cardinalities': cardinalities
      }, f)
  ```
- **Inference function**: Load artifacts and apply same transformations to new events

---

## Complete Example: OCSF to Model Input

```{code-cell}
:tags: [skip-execution]

# Step 1: Load OCSF events
events = load_ocsf_from_file('ocsf_events.jsonl')
print(f"Loaded {len(events)} events")

# Step 2: Extract features
extractor = OCSFFeatureExtractor(window_minutes=60)
df = extractor.batch_extract(events)
print(f"Extracted features shape: {df.shape}")

# Step 3: Prepare for TabularResNet
numerical, categorical, encoders, scaler, cardinalities = prepare_for_tabular_resnet(
    df,
    extractor.categorical_features,
    extractor.numerical_features
)

print(f"\nReady for TabularResNet:")
print(f"  Numerical features: {numerical.shape}")
print(f"  Categorical features: {categorical.shape}")
print(f"  Categorical cardinalities: {cardinalities}")

# Step 4: Create TabularResNet model (from Part 2)
# model = TabularResNet(
#     num_numerical_features=numerical.shape[1],
#     categorical_cardinalities=cardinalities,
#     d_model=256,
#     num_blocks=6,
#     num_classes=None  # Embedding mode
# )

# Step 5: Train (see Part 4 - Self-Supervised Training)
```

---

## Summary

In this part, you learned:

1. **OCSF structure**: Nested JSON with 300+ optional fields
2. **Feature extraction**: Flattening, core features, temporal patterns
3. **Feature engineering**: Derived features, aggregations, cyclical encoding
4. **Data challenges**: Missing values, high cardinality, sparse fields
5. **End-to-end pipeline**: OCSF JSON → DataFrame → TabularResNet arrays

**Key takeaways**:
- Not all OCSF fields are useful - start with 20-50 core features
- Temporal features (hour, day) and aggregations (event counts) are critical for anomaly detection
- Cyclical encoding (sin/cos) preserves circular nature of time
- Hashing trick handles high-cardinality fields (IPs, user IDs)
- Missing values need explicit handling (special category or indicator)

**Next**: In [Part 4](part4-self-supervised-training), you'll use these engineered features to train TabularResNet with self-supervised learning on unlabelled OCSF data.

---

## References

```{bibliography}
:filter: docname in docnames
:style: unsrt
```

from flask import Flask, request, jsonify, Response
import time
import random
import logging
import json
import os
import socket
import uuid
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Service identity
SERVICE_NAME = "web-api"
SERVICE_VERSION = "1.0.0"
HOSTNAME = socket.gethostname()

# Simulated users for realistic data
SIMULATED_USERS = [
    {"uid": "user-1001", "name": "alice.johnson", "email": "alice@company.com", "department": "engineering"},
    {"uid": "user-1002", "name": "bob.smith", "email": "bob@company.com", "department": "sales"},
    {"uid": "user-1003", "name": "carol.williams", "email": "carol@company.com", "department": "support"},
    {"uid": "user-1004", "name": "david.brown", "email": "david@company.com", "department": "marketing"},
    {"uid": "user-1005", "name": "eve.davis", "email": "eve@company.com", "department": "engineering"},
]

# OpenTelemetry setup for unified observability (logs, traces, metrics via OTLP)
otel_logger = None
tracer = None
otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector:4317')

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME as RESOURCE_SERVICE_NAME
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    # Create shared resource
    resource = Resource.create({
        RESOURCE_SERVICE_NAME: SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "host.name": HOSTNAME,
    })

    # Set up tracing
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    otlp_trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_trace_exporter))

    # Set up logging via OTLP
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    otlp_log_exporter = OTLPLogExporter(endpoint=otlp_endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))

    # Create OTel logging handler
    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    otel_logger = logging.getLogger("otel." + SERVICE_NAME)
    otel_logger.setLevel(logging.INFO)
    otel_logger.addHandler(otel_handler)

    print(f"[{SERVICE_NAME}] OpenTelemetry enabled: logs and traces via OTLP to {otlp_endpoint}")
except Exception as e:
    print(f"[{SERVICE_NAME}] OpenTelemetry setup failed: {e}")
    otel_logger = None
    tracer = None


class StructuredLogger:
    """Logger that emits OCSF-ready structured JSON logs via OpenTelemetry."""

    def __init__(self, service_name):
        self.service_name = service_name
        # Fallback to stdout if OTel unavailable
        self.fallback_logger = logging.getLogger(service_name + ".fallback")
        self.fallback_logger.setLevel(logging.INFO)
        self.fallback_logger.handlers = []
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.fallback_logger.addHandler(handler)
        self.fallback_logger.propagate = False

    def _emit(self, level, message, **kwargs):
        """Emit a structured log entry with OCSF-compatible fields."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        log_entry = {
            "timestamp": timestamp,
            "time": int(time.time() * 1000),
            "service": self.service_name,
            "level": level,
            "message": message,
            "metadata": {
                "version": "1.0.0",
                "product": {
                    "name": self.service_name,
                    "version": SERVICE_VERSION,
                    "vendor_name": "Demo"
                }
            },
            "device": {
                "hostname": HOSTNAME,
                "type": "server",
                "type_id": 1
            }
        }

        # Add optional OCSF fields
        for key, value in kwargs.items():
            if value is not None:
                log_entry[key] = value

        # Log via OpenTelemetry if available (JSON in message body)
        if otel_logger:
            log_method = getattr(otel_logger, level.lower(), otel_logger.info)
            log_method(json.dumps(log_entry))

        # Always log to stdout for Docker capture (backup)
        self.fallback_logger.info(json.dumps(log_entry))

    def info(self, message, **kwargs):
        self._emit("INFO", message, **kwargs)

    def warning(self, message, **kwargs):
        self._emit("WARNING", message, **kwargs)

    def error(self, message, **kwargs):
        self._emit("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self._emit("DEBUG", message, **kwargs)


# Initialize structured logger
logger = StructuredLogger(SERVICE_NAME)

# Disable verbose werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', ['query_type'])

# Database and cache connections (lazy initialization)
db_conn = None
cache = None

def get_db():
    global db_conn
    if db_conn is None:
        try:
            import psycopg2
            db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@postgres:5432/appdb')
            db_conn = psycopg2.connect(db_url)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable: {e}")
    return db_conn

def get_cache():
    global cache
    if cache is None:
        try:
            import redis as redis_lib
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
            cache = redis_lib.from_url(redis_url, decode_responses=True)
            cache.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
    return cache


def get_request_context():
    """Extract request context for structured logging."""
    # Simulate user from request (in production, this would come from auth)
    user = random.choice(SIMULATED_USERS)
    session_id = f"sess-{uuid.uuid4().hex[:12]}"

    # Get client IP (may be forwarded)
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip and ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    return {
        "actor": {
            "user": {
                "uid": user["uid"],
                "name": user["name"],
                "email": user["email"]
            },
            "session": {
                "uid": session_id,
                "created_time": int(time.time() * 1000)
            }
        },
        "src_endpoint": {
            "ip": client_ip or "192.168.1.100",
            "port": random.randint(30000, 65000),
            "domain": request.headers.get('Host', 'unknown')
        },
        "dst_endpoint": {
            "ip": "10.0.0.1",
            "port": 8000,
            "svc_name": SERVICE_NAME
        },
        "http_request": {
            "method": request.method,
            "url": {
                "path": request.path,
                "query_string": request.query_string.decode() if request.query_string else "",
                "scheme": request.scheme,
                "hostname": request.host
            },
            "user_agent": request.headers.get('User-Agent', 'unknown'),
            "http_headers": dict(list(request.headers)[:5])  # First 5 headers
        }
    }


def log_api_activity(message, status_code, duration_ms, trace_id=None, activity_id=1, **extra):
    """Log an API activity event with full OCSF context."""
    ctx = get_request_context()

    # Determine status
    if status_code >= 500:
        status_id = 2  # Failure
        severity_id = 4  # Error
    elif status_code >= 400:
        status_id = 2  # Failure
        severity_id = 3  # Warning
    else:
        status_id = 1  # Success
        severity_id = 2  # Info

    # Calculate type_uid: class_uid * 100 + activity_id
    class_uid = 6003  # API Activity
    type_uid = class_uid * 100 + activity_id

    logger.info(
        message,
        class_uid=class_uid,
        class_name="API Activity",
        category_uid=6,
        category_name="Application Activity",
        activity_id=activity_id,
        activity_name=["Unknown", "Create", "Read", "Update", "Delete"][min(activity_id, 4)],
        type_uid=type_uid,
        severity_id=severity_id,
        status_id=status_id,
        status_code=str(status_code),
        status=["Unknown", "Success", "Failure"][status_id],
        duration=duration_ms,
        trace_id=str(trace_id) if trace_id else None,
        actor=ctx["actor"],
        src_endpoint=ctx["src_endpoint"],
        dst_endpoint=ctx["dst_endpoint"],
        http_request=ctx["http_request"],
        http_response={
            "code": status_code,
            "status": "OK" if status_code < 400 else "Error",
            "latency": duration_ms
        },
        **extra
    )


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Fetch user data - normal operation."""
    start_time = time.time()
    trace_id = None

    # Start span if tracing available
    span = None
    if tracer:
        span = tracer.start_span("get_user")
        span.set_attribute("user.id", user_id)
        trace_id = span.get_span_context().trace_id

    try:
        # Check cache first
        redis_cache = get_cache()
        if redis_cache:
            try:
                cached = redis_cache.get(f"user:{user_id}")
                if cached:
                    duration_ms = (time.time() - start_time) * 1000
                    request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                    request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                    log_api_activity(
                        f"Cache hit for user {user_id}",
                        status_code=200,
                        duration_ms=duration_ms,
                        trace_id=trace_id,
                        activity_id=2,  # Read
                        resources=[{"type": "user", "uid": str(user_id), "data": {"source": "cache"}}]
                    )
                    return jsonify({"user_id": user_id, "source": "cache"})
            except Exception as e:
                logger.warning(f"Cache error: {e}")

        # Query database
        conn = get_db()
        if conn:
            try:
                db_start = time.time()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                result = cursor.fetchone()
                db_query_duration.labels(query_type='SELECT').observe(time.time() - db_start)

                if result:
                    if redis_cache:
                        redis_cache.setex(f"user:{user_id}", 300, str(result))
                    duration_ms = (time.time() - start_time) * 1000
                    request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
                    request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
                    log_api_activity(
                        f"User {user_id} fetched from database",
                        status_code=200,
                        duration_ms=duration_ms,
                        trace_id=trace_id,
                        activity_id=2,  # Read
                        resources=[{"type": "user", "uid": str(user_id), "data": {"source": "database"}}]
                    )
                    return jsonify({"user_id": user_id, "source": "database"})
            except Exception as e:
                logger.warning(f"Database error: {e}")

        # User not found or DB unavailable - return mock data for demo
        duration_ms = (time.time() - start_time) * 1000
        request_count.labels(method='GET', endpoint='/api/users', status=200).inc()
        request_duration.labels(method='GET', endpoint='/api/users').observe(time.time() - start_time)
        log_api_activity(
            f"Returning mock user {user_id}",
            status_code=200,
            duration_ms=duration_ms,
            trace_id=trace_id,
            activity_id=2,  # Read
            resources=[{"type": "user", "uid": str(user_id), "data": {"source": "mock"}}]
        )
        return jsonify({"user_id": user_id, "source": "mock", "name": f"User {user_id}"})

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        request_count.labels(method='GET', endpoint='/api/users', status=500).inc()
        log_api_activity(
            f"Error fetching user {user_id}: {str(e)}",
            status_code=500,
            duration_ms=duration_ms,
            trace_id=trace_id,
            activity_id=2,
            error={"message": str(e), "type": type(e).__name__}
        )
        return jsonify({"error": "Internal server error"}), 500

    finally:
        if span:
            span.end()


@app.route('/api/checkout', methods=['POST'])
def checkout():
    """Checkout operation - can trigger anomalies."""
    start_time = time.time()
    trace_id = None

    span = None
    if tracer:
        span = tracer.start_span("checkout")
        trace_id = span.get_span_context().trace_id

    try:
        # Simulate anomalies based on environment variable
        anomaly_prob = float(os.getenv('ANOMALY_PROBABILITY', 0.05))

        if random.random() < anomaly_prob:
            # ANOMALY: Simulate various failure modes
            anomaly_type = random.choice(['db_timeout', 'memory_leak', 'slow_query', 'cache_miss_storm'])

            if anomaly_type == 'db_timeout':
                time.sleep(5)  # Simulate slow query
                duration_ms = (time.time() - start_time) * 1000
                request_count.labels(method='POST', endpoint='/api/checkout', status=504).inc()
                log_api_activity(
                    "Database timeout during checkout",
                    status_code=504,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    activity_id=1,  # Create
                    anomaly={"type": "db_timeout", "severity": "high"},
                    error={"message": "Database timeout", "type": "TimeoutError"}
                )
                return jsonify({"error": "Database timeout"}), 504

            elif anomaly_type == 'memory_leak':
                # Simulate memory leak by holding large objects
                leak = ["x" * 1000000 for _ in range(100)]  # 100MB allocation
                duration_ms = (time.time() - start_time) * 1000
                log_api_activity(
                    "High memory allocation during checkout",
                    status_code=200,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    activity_id=1,
                    anomaly={"type": "memory_leak", "severity": "medium", "memory_mb": 100}
                )

            elif anomaly_type == 'slow_query':
                time.sleep(random.uniform(2, 5))
                duration_ms = (time.time() - start_time) * 1000
                log_api_activity(
                    f"Slow checkout processing: {duration_ms:.0f}ms",
                    status_code=200,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    activity_id=1,
                    anomaly={"type": "slow_query", "severity": "medium", "threshold_ms": 2000}
                )

            elif anomaly_type == 'cache_miss_storm':
                # Simulate cache invalidation causing DB overload
                redis_cache = get_cache()
                if redis_cache:
                    for i in range(50):
                        redis_cache.delete(f"user:{i}")
                duration_ms = (time.time() - start_time) * 1000
                log_api_activity(
                    "Cache miss storm detected",
                    status_code=200,
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                    activity_id=1,
                    anomaly={"type": "cache_miss_storm", "severity": "high", "invalidated_keys": 50}
                )

        # Normal checkout flow
        duration_ms = (time.time() - start_time) * 1000
        request_count.labels(method='POST', endpoint='/api/checkout', status=200).inc()
        request_duration.labels(method='POST', endpoint='/api/checkout').observe(time.time() - start_time)
        log_api_activity(
            "Checkout completed successfully",
            status_code=200,
            duration_ms=duration_ms,
            trace_id=trace_id,
            activity_id=1,  # Create
            resources=[{"type": "order", "uid": f"order-{uuid.uuid4().hex[:8]}"}]
        )
        return jsonify({"status": "success"})

    finally:
        if span:
            span.end()


@app.route('/api/search', methods=['GET'])
def search():
    """Search endpoint for load generator."""
    start_time = time.time()
    query = request.args.get('q', '')

    # Simulate search delay
    time.sleep(random.uniform(0.01, 0.1))

    duration_ms = (time.time() - start_time) * 1000
    request_count.labels(method='GET', endpoint='/api/search', status=200).inc()
    request_duration.labels(method='GET', endpoint='/api/search').observe(time.time() - start_time)
    log_api_activity(
        f"Search completed for query: {query}",
        status_code=200,
        duration_ms=duration_ms,
        activity_id=2,  # Read
        resources=[{"type": "search", "data": {"query": query, "results_count": 0}}]
    )

    return jsonify({"results": [], "query": query})


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    logger.info("Starting web-api service on port 8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)

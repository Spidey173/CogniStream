"""
Prometheus metrics telemetry definitions for streaming vision system.
"""

from prometheus_client import Counter, Gauge, Histogram

# HTTP & API Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled",
    ["method", "endpoint", "status_code"],
)

# Stream & Connection Metrics
ACTIVE_WS_CONNECTIONS = Gauge(
    "active_websocket_connections",
    "Current active camera ingestion and viewer connections",
    ["type", "camera_id"],
)

FRAMES_PROCESSED_TOTAL = Counter(
    "frames_processed_total",
    "Total frames ingested and processed",
    ["camera_id", "detector"],
)

FRAME_PROCESSING_LATENCY = Histogram(
    "frame_processing_latency_seconds",
    "End-to-end frame ingestion and vision pipeline processing latency",
    ["camera_id", "stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

FPS_GAUGE = Gauge(
    "camera_fps",
    "Current processing frame rate per second",
    ["camera_id"],
)

TRACKED_FACES_GAUGE = Gauge(
    "tracked_faces_current",
    "Currently tracked face entities in active camera stream",
    ["camera_id"],
)

# Database & Storage Metrics
DB_BATCH_INSERT_LATENCY = Histogram(
    "db_batch_insert_latency_seconds",
    "PostgreSQL ROI batch insertion latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

# Infrastructure Health Gauges
REDIS_CONNECTED_GAUGE = Gauge(
    "redis_connected",
    "Redis Pub/Sub connection status (1 = connected, 0 = disconnected)",
)

POSTGRES_CONNECTED_GAUGE = Gauge(
    "postgres_connected",
    "PostgreSQL connection status (1 = connected, 0 = disconnected)",
)

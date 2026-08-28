# 🧠 CogniStream — Cognitive Real-Time Vision Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular real-time computer vision platform demonstrating distributed streaming, event-driven processing, and pluggable vision backends.

Built with **FastAPI**, **MediaPipe**, **Redis Pub/Sub**, **SQLAlchemy Async**, **PostgreSQL**, and **React**.

Designed around **Domain-Driven Design (DDD)** principles, **MessageBus/EventBus decoupling**, **Pluggable Vision Detectors**, **Centroid Entity Tracking**, and **Async ROI Database Batching**.

---

## 🏗️ Architecture Overview

```text
 ┌──────────────────────┐   Frame    ┌────────────────────────────────────────────────────────┐   Event    ┌───────────────┐
 │ BaseCameraSource     │───────────►│ VisionPipeline                                         │───────────►│ EventBus      │
 │ (Browser/File/Mock)  │            │ (Decode ──► Detect ──► Track ──► Annotate ──► Publish) │            │ (Events)      │
 └──────────┬───────────┘            └───────────────────────────┬────────────────────────────┘            └───────┬───────┘
            │                                                    │                                                 │
      CameraRegistry                                         MessageBus (Frames)                                   │
            │                                                    │                                                 ▼
            ▼                                                    ▼                                       Async ROI Batch Writer
 /api/v1/cameras                                         /video/{camera_id}                                        │
 /api/v1/stats                                           /snapshot/{camera_id}                                     ▼
                                                                                                               PostgreSQL
```

### System Component Breakdown

1. **Domain Layer (`app/domain/`)**: Explicit business models representing core vision concepts (`Frame`, `DetectionResult`, `BoundingBox`, `TrackedEntity`, `CameraInfo`).
2. **Vision Processing Pipeline (`DefaultVisionPipeline`)**: Decoupled, sequential execution stages (`DecodeStage` → `DetectStage` → `TrackStage` → `AnnotateStage` → `PublishStage`).
3. **Pluggable Detector Backends (`BaseDetector`)**: Abstract interface exposing detector metadata (`name`, `version`, `backend`, `supports_gpu`, `latency_ms`) with plugin registration (`register_detector`).
4. **Real-Time Entity Tracking (`CentroidTracker`)**: Assigns persistent tracking IDs (`track_id`) across successive video frames.
5. **Decoupled MessageBus & EventBus**:
   - `MessageBus`: High-throughput binary JPEG frame distribution across Redis Pub/Sub channels (enabling multi-instance FastAPI horizontal scaling behind Nginx load balancers).
   - `EventBus`: Structured JSON telemetry events for background analytics.
6. **Async ROI DB Batch Writer (`ROIBatchProcessor`)**: Accumulates detection events in an async queue and executes bulk PostgreSQL insertions (`insert_roi_batch`), eliminating per-frame database write bottlenecks.

---

## 🔍 Pluggable Detector Backends Comparison

| Detector Key | Detector Name | Detection Range | Avg Latency | Target Use Case |
| :--- | :--- | :--- | :---: | :--- |
| `mediapipe` | MediaPipe BlazeFace (Short-Range) | `< 2 meters` | `~6-8 ms` | Ultra-fast short-range webcam / selfie stream processing |
| `mediapipe-full` | MediaPipe BlazeFace (Full-Range) | `< 5 meters` | `~12-16 ms` | Multi-scale anchor boxes for wider field-of-view |
| `mock` | Synthetic Mock Detector | N/A | `~1 ms` | High-speed synthetic detector for unit testing and CI |

Switch detector backend via environment variable:
```bash
DETECTOR_TYPE=mediapipe-full
```

---

## 🚀 Quick Start (≤ 3 minutes)

### Prerequisites

- **Python 3.11+** and **Node.js 18+** (for local development)
- **Docker & Docker Compose** (optional for containerized deployment)

---

### Option A: Local Execution (Without Docker)

#### 1. Start FastAPI Backend Server
```bash
DATABASE_URL=sqlite+aiosqlite:///./facedetect.db python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*(If `python3` points to a system Python version without packages, use `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`).*

#### 2. Start React Frontend UI
In a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```

---

### Option B: Containerized Execution with Docker Compose

```bash
docker compose up --build -d
```

---

### Verify System Endpoints

- **Frontend Application** → [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Docs** → [http://localhost:8000/docs](http://localhost:8000/docs)
- **Liveness Probe (`/health`)** → [http://localhost:8000/health](http://localhost:8000/health)
- **Readiness Probe (`/ready`)** → [http://localhost:8000/ready](http://localhost:8000/ready)
- **Prometheus Metrics (`/metrics`)** → [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Camera Telemetry (`/api/v1/stats`)** → [http://localhost:8000/api/v1/stats](http://localhost:8000/api/v1/stats)

---

## ⚖️ Architectural Design Trade-Offs

| Decision | Selection | Alternative | Rationale |
| :--- | :--- | :--- | :--- |
| **Stream Transport** | **MJPEG over HTTP** | WebRTC / RTSP | MJPEG requires zero signaling servers or complex WebRTC ICE/STUN handshakes, plays natively inside browser `<img>` tags, and offers simple server-to-client video broadcasting. |
| **Inter-Node Bus** | **Redis Pub/Sub** | Apache Kafka / NATS | Redis Pub/Sub delivers sub-millisecond memory frame broadcasting with minimal operational overhead for multi-replica FastAPI nodes. (Future evolution: Redis Streams or Kafka for durable event replay). |
| **Persistence Strategy** | **Async Batch Buffer** | Direct per-frame INSERT | Writing 30 DB rows/sec per camera locks connection pools. `ROIBatchProcessor` queues detection events and flushes bulk batches every 50 items or 2.0s. |
| **Bounding Box Rendering** | **NumPy Array Slicing** | OpenCV (`cv2`) | In-place NumPy matrix slicing modifies frame arrays in ~0.05ms without intermediate object allocations or heavy OpenCV C++ library dependencies. |

---

## 📊 End-to-End Load Benchmarking

The platform includes a built-in multi-camera load benchmarking script (`scripts/benchmark.py`) to measure system throughput, end-to-end frame latency, and ACK rates under load.

### Run Benchmark Script

```bash
python scripts/benchmark.py --url ws://localhost:8000/api/v1/stream --cameras 4 --fps 30 --duration 10
```

### Sample Performance Metrics

```json
{
  "timestamp": "2026-08-07T00:00:00Z",
  "total_cameras": 4,
  "aggregate_fps": 118.5,
  "average_latency_ms": 8.4,
  "total_frames_processed": 1185
}
```

---

## 🔌 API Reference Highlights

### Streaming & Ingestion

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/stream/{camera_id}` | `WS` | Ingest binary JPEG video frames over WebSocket (`?api_key=...`). |
| `/api/v1/video/{camera_id}` | `GET` | Stream live processed video feed (MJPEG). |
| `/api/v1/snapshot/{camera_id}`| `GET` | Get latest processed JPEG snapshot frame. |

### Telemetry & Querying

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/stats` | `GET` | System metrics (active streams, total viewers, FPS, detector info). |
| `/api/v1/cameras` | `GET` | List active camera sources. |
| `/api/v1/detectors` | `GET` | List pluggable vision detector backends. |
| `/api/v1/roi` | `GET` | Query paginated face detection ROI records (`limit`, `offset`, `camera_id`). |

---

## 🧪 Testing & Code Quality

Run automated test suite with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

Run code formatting and lint checks:

```bash
black --check app tests
ruff check app tests
mypy app
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Runtime environment (`development`, `testing`, `production`) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL for Pub/Sub frame bus |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL database connection string |
| `API_KEY` | `dev-secret-api-key` | Security API Key / Bearer token |
| `DETECTOR_TYPE` | `mediapipe` | Vision detector backend (`mediapipe`, `mediapipe-full`, `mock`) |
| `BATCH_SIZE` | `50` | ROI batch flush threshold |
| `BATCH_FLUSH_INTERVAL`| `2.0` | ROI batch flush interval in seconds |
| `MAX_FRAME_SIZE_BYTES`| `5242880` | Maximum incoming frame payload size limit (5MB) |

---

## 📄 License

Distributed under the MIT License.

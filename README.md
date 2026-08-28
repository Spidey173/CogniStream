# 🧠 CogniStream — Cognitive Real-Time Vision Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance real-time computer vision platform featuring distributed video streaming, event-driven processing, and pluggable vision backends.

Built with **FastAPI**, **MediaPipe**, **OpenCV**, **SQLAlchemy Async**, **Neon / PostgreSQL**, and **React (Vite)**.

Designed around **Domain-Driven Design (DDD)** principles, **MessageBus/EventBus decoupling**, **Centroid Entity Tracking**, and **Async ROI Database Batching**.

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
2. **Vision Processing Pipeline (`DefaultVisionPipeline`)**: Decoupled, sequential execution stages (`DecodeStage` → `DetectStage` → `TrackStage` → `AnnotateStage` → `PublishStage`) powered by fast C-level OpenCV acceleration.
3. **Pluggable Detector Backends (`BaseDetector`)**: Pluggable face detection interface with dynamic model loading and automated fallback.
4. **Real-Time Entity Tracking (`CentroidTracker`)**: Assigns persistent tracking IDs (`track_id`) across successive video frames.
5. **Decoupled MessageBus & EventBus**:
   - `MessageBus`: High-throughput binary JPEG frame distribution across in-memory queues or Redis Pub/Sub channels.
   - `EventBus`: Structured JSON telemetry events for background analytics.
6. **Async ROI DB Batch Writer (`ROIBatchProcessor`)**: Accumulates detection events in an async queue and executes bulk PostgreSQL insertions (`insert_roi_batch`), eliminating per-frame database write bottlenecks.

---

## 🔍 Pluggable Detector Backends Comparison

| Detector Key | Detector Name | Detection Range | Avg Latency | Target Use Case |
| :--- | :--- | :--- | :---: | :--- |
| `mediapipe` | MediaPipe BlazeFace (Short-Range) | `< 2 meters` | `~5-8 ms` | Ultra-fast short-range webcam / selfie stream processing |
| `mediapipe-full` | MediaPipe BlazeFace (Full-Range) | `< 5 meters` | `~10-15 ms` | Multi-scale anchor boxes for wider field-of-view |
| `mock` | Synthetic Mock Detector | N/A | `~1 ms` | High-speed synthetic detector for unit testing and CI |

Switch detector backend via environment variable:
```bash
DETECTOR_TYPE=mediapipe
```

---

## 🚀 Quick Start & Local Execution

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**

---

### 1. Start FastAPI Backend Server

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Uvicorn server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### 2. Start React Frontend UI

In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

---

## ☁️ Cloud Deployment

### Backend on Render (Native Python)
1. In Render, create a **New Web Service** pointing to your repository.
2. Select **Python 3** environment:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
3. Set environment variables:
   - `ENVIRONMENT`: `production`
   - `DATABASE_URL`: `postgresql://<user>:<password>@<neon-host>/neondb?sslmode=require`
   - `API_KEY`: `dev-secret-api-key`
   - `ALLOWED_ORIGINS`: `*`

### Frontend on Vercel
1. In Vercel, import your repository:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
2. Add Environment Variable:
   - `VITE_API_URL`: `https://<YOUR-RENDER-BACKEND-URL>.onrender.com`
3. Click **Deploy**.

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
| `/api/v1/roi/latest` | `GET` | Query latest face detection ROI records (`count=5`). |
| `/api/v1/roi` | `GET` | Query paginated face detection history (`limit`, `offset`, `camera_id`). |

### Health Probes

- **Liveness (`/health`)** → Returns `{"status": "healthy"}`
- **Readiness (`/ready`)** → Returns database and message bus health
- **Metrics (`/metrics`)** → Prometheus scrape endpoint

---

## 🧪 Testing & Code Quality

Run automated test suite:
```bash
pytest
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Runtime environment (`development`, `testing`, `production`) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL database connection string (Neon / Render) |
| `API_KEY` | `dev-secret-api-key` | Security API Key / Bearer token |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `DETECTOR_TYPE` | `mediapipe` | Vision detector backend (`mediapipe`, `mediapipe-full`, `mock`) |
| `BATCH_SIZE` | `50` | ROI batch flush threshold |
| `BATCH_FLUSH_INTERVAL`| `2.0` | ROI batch flush interval in seconds |
| `MAX_FRAME_SIZE_BYTES`| `5242880` | Maximum incoming frame payload size limit (5MB) |

---

## 📄 License

Distributed under the MIT License.

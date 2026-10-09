# 🧠 CogniStream — Real-Time Edge Vision & Affect Analytics

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![CI Pipeline](https://github.com/Spidey173/CogniStream/actions/workflows/ci.yml/badge.svg)](https://github.com/Spidey173/CogniStream/actions/workflows/ci.yml)
[![Platform: Vercel](https://img.shields.io/badge/Deploy-Vercel-black.svg)](https://vercel.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time computer vision and affective computing platform combining **client-side WebAssembly inference**, **Facial Action Coding System (FACS) feature extraction**, and **serverless telemetry persistence (FastAPI + Neon PostgreSQL)**.

**🚀 Live Production App:** [https://cognistream17.vercel.app/](https://cognistream17.vercel.app/)

---

## 💡 System Architecture & Engineering Decisions

```text
 ┌────────────────────────────────────────────────────────┐
 │  Client Browser (WebAssembly / WebGL)                  │
 │  - Webcam Ingestion via getUserMedia (640×480 @ 30 FPS)│
 │  - Face Detection & 68-Point Craniofacial Landmarks    │
 │  - Action Units: Zygomatic Smile, EAR Openness, Brow   │
 │  - 50-State Cognitive & Emotion Taxonomy Evaluation    │
 │  - Centroid Entity Tracking with IoU Association       │
 │  - Real-Time HTML5 Canvas HUD (~30-60 FPS local loop)  │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            │ Asynchronous Batch Sync (~3s debounce)
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Vercel Serverless API (/api/index.py)                 │
 │  - FastAPI lightweight serverless route handler        │
 │  - Ingests coordinate & affect metadata (no raw video) │
 │  - In-memory ring buffer fallback if DB is unreachable │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Neon Serverless PostgreSQL Database                   │
 │  - Relational persistence for spatial telemetry        │
 │  - Queryable audit history for session analytics       │
 └────────────────────────────────────────────────────────┘
```

### Why Process on the Client Edge?

* **Zero Video Streaming Bandwidth:** Traditional CV pipelines stream heavy MJPEG or WebRTC video frames to servers, causing bandwidth strain and cloud egress costs. CogniStream executes landmark tracking directly in the user's browser, sending only lightweight JSON coordinates and classification stats (`< 1 KB/batch`).
* **Privacy-Preserving by Design:** Camera frames never leave the user's device memory. Only aggregated metric numbers are persisted to the cloud database.
* **Predictable Cloud Costs:** By offloading computer vision compute to client WebAssembly/GPU runtimes, the backend remains a lightweight serverless CRUD API that fits inside free serverless tiers.

---

## 🔬 Core Technical Modules

### 1. 68-Point Facial Action Units (FAU)
The edge engine computes heuristic geometric metrics inspired by the **Facial Action Coding System (FACS)**:
* **Zygomaticus Major (Smile Intensity):** Mouth corner elevation normalized against jaw width.
* **Eye Aspect Ratio (EAR / Ocular Aperture):** Vertical-to-horizontal eyelid distance ratio to measure blink rate and attention alertness.
* **Corrugator Supercilii (Brow Strain):** Inter-eyebrow contraction distance measuring cognitive strain vs relaxation.
* **3D Cranial Pose:** Estimated Euler angles (Yaw, Pitch, Roll) computed from craniofacial landmark symmetry vectors.

### 2. 50-State Cognitive & Affective Taxonomy
Rather than relying solely on 7 macro classes, the app maps combinations of facial action units, 3D pose, and base probabilities to a **50-state granular taxonomy** (e.g., *Analytical Focus*, *Inquisitive Interest*, *Subtle Smirk*, *Fatigued Strain*).

### 3. Russell's Circumplex Affect Space
Biometric readings are translated into continuous **Valence (Pleasure)** and **Arousal (Activation)** dimensions, plotting real-time psychological states onto a 2D coordinate plane.

### 4. Centroid Entity Tracking
To maintain stable identity IDs across frames, the system calculates Euclidean centroid distances between sequential bounding boxes, applying exponential moving averages (EMA) to prevent label flickering.

---

## 📊 Client vs Cloud Pipeline Comparison

| Consideration | Server-Side Streaming (MJPEG/WebRTC) | CogniStream Client-Edge Architecture |
| :--- | :--- | :--- |
| **Inference Compute** | Centralized GPU servers (AWS EC2 / GCP) | Client CPU/GPU via WebAssembly SIMD |
| **Video Bandwidth** | Continuous upstream upload (~1.5–5 Mbps) | **0 Kbps** (video stays in local browser memory) |
| **Telemetry Payload** | Raw image frames | Compact JSON vectors (`< 1 KB` per batch) |
| **User Privacy** | Video processed on external servers | Video strictly retained client-side |
| **Backend Cost** | High continuous VM/GPU hosting fees | Low-cost Serverless Function (`fastapi`) |

---

## 🛠️ Tech Stack

* **Frontend:** React 19, Vite, Canvas 2D API, Custom CSS Design System (Dark Red Magma theme)
* **Vision & ML:** Face-API.js / WebAssembly, 68-Point Facial Landmark Net, MobileNet-based TinyFace
* **Backend:** Python 3.11, FastAPI, SQLAlchemy (AsyncIO), Pydantic v2
* **Storage & Persistence:** Neon Serverless PostgreSQL (`asyncpg`)
* **Deployment & CI:** Vercel Serverless Functions, GitHub Actions CI Pipeline

---

## 🚀 Local Development Setup

### Prerequisites
* **Node.js 18+** & **npm**
* **Python 3.11+**

### 1. Run Frontend Application
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 2. (Optional) Run Python Serverless API Locally
```bash
pip install -r requirements.txt
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Backend Test Suite
```bash
pip install -r requirements.txt -r requirements-test.txt
pytest
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health probe reporting runtime status and database connectivity. |
| `/api/stats` | `GET` | Returns telemetry cache counts and active configuration. |
| `/api/roi` | `POST` | Ingests batched face coordinates and affect metadata. |
| `/api/roi/latest` | `GET` | Queries recent detection logs for dashboard auditing (`?count=10`). |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


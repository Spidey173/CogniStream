# 🚀 Deploying CogniStream AI to Render

This guide provides step-by-step instructions to deploy the **CogniStream AI** real-time vision platform to [Render](https://render.com).

---

## 📋 Overview of Deployment Architecture

| Component | Render Service Type | Details |
| :--- | :--- | :--- |
| **Backend** | **Web Service (Docker)** | FastAPI app running MediaPipe face detection, WebSockets, and MJPEG video delivery. |
| **Frontend** | **Static Site** | React + Vite UI deployed to Render's global CDN. |
| **Database** | **PostgreSQL (Neon or Render)** | Stores detection ROI records and analytics. |

---

## 🌟 Method 1: 1-Click Blueprint Deployment (Fastest)

CogniStream includes a [`render.yaml`](./render.yaml) Blueprint that configures the entire stack automatically.

### Steps:
1. **Push your repository** to GitHub or GitLab:
   ```bash
   git add .
   git commit -m "Configure Render deployment"
   git push origin main
   ```
2. Log in to your [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** in the top navigation bar and select **Blueprint**.
4. Connect your **CogniStream repository**.
5. Render will automatically detect [`render.yaml`](./render.yaml) and list the services to create:
   - `cognistream-backend` (Docker Web Service)
   - `cognistream-frontend` (Static Site)
   - `cognistream-db` (Managed PostgreSQL)
6. *(Optional)* If using an external database like **Neon**:
   - In the Blueprint setup page or in the Backend Service settings under **Environment Variables**, set `DATABASE_URL` to your Neon connection string:
     ```text
     postgresql://neondb_owner:npg_AGs3zgUYIqm8@ep-cold-union-ae0qi9d0-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require
     ```
7. Click **Apply**. Render will automatically build the container and deploy both services!

---

## 🛠️ Method 2: Step-by-Step Manual Setup

If you prefer configuring each service individually in the Render UI:

### Step 1: Set Up Database (Neon or Render PostgreSQL)

#### Option A: Using your existing Neon Database
You can directly use your connection string:
```text
postgresql://neondb_owner:npg_AGs3zgUYIqm8@ep-cold-union-ae0qi9d0-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require
```

#### Option B: Using Render Managed PostgreSQL
1. On Render Dashboard, click **New +** $\rightarrow$ **PostgreSQL**.
2. **Name**: `cognistream-db`
3. **Database**: `facedetect`
4. **User**: `postgres`
5. **Plan**: Free (or Starter).
6. Click **Create Database** and copy the **Internal Database URL** (or External URL).

---

### Step 2: Deploy Backend Web Service

1. On Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Connect your Git repository.
3. Configure the following settings:
   - **Name**: `cognistream-backend`
   - **Language / Runtime**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: Free (or Starter for higher CPU)
   - **Health Check Path**: `/health`
4. Under **Environment Variables**, add:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `ENVIRONMENT` | `production` | Enables production logging and security checks |
   | `DATABASE_URL` | *Your Neon or Render PostgreSQL URL* | Database connection string |
   | `API_KEY` | `dev-secret-api-key` *(or your custom key)* | API Key for streaming & stats |
   | `ALLOWED_ORIGINS` | `*` *(or your frontend URL)* | CORS allowed origins |
   | `BATCH_SIZE` | `50` | ROI batch write size |
   | `BATCH_FLUSH_INTERVAL`| `2.0` | ROI batch flush timer (seconds) |
   | `DETECTION_CONFIDENCE`| `0.5` | Minimum face detection confidence |
   | `DETECTOR_TYPE` | `mediapipe` | Pluggable detector backend |
5. Click **Create Web Service**.
6. Wait for the build to finish. Once live, copy your backend URL (e.g. `https://cognistream-backend.onrender.com`).

---

### Step 3: Deploy Frontend Static Site

1. On Render Dashboard, click **New +** $\rightarrow$ **Static Site**.
2. Connect your Git repository.
3. Configure the following settings:
   - **Name**: `cognistream-frontend`
   - **Branch**: `main`
   - **Root Directory**: *(leave blank or `frontend`)*
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`
4. Under **Redirects / Rewrites**, add an SPA fallback rewrite rule:
   - **Type**: `Rewrite`
   - **Source**: `/*`
   - **Destination**: `/index.html`
5. Under **Environment Variables**, add:
   | Key | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://cognistream-backend.onrender.com` *(your backend URL from Step 2)* |
   | `VITE_API_KEY` | `dev-secret-api-key` *(matching backend API_KEY)* |
6. Click **Create Static Site**.

---

## 🔒 HTTPS & Webcam Permissions

- Browsers strictly enforce that camera access (`navigator.mediaDevices.getUserMedia`) is only accessible over **HTTPS** (or localhost).
- Render automatically provisions free SSL/TLS certificates for both your backend and frontend (`https://*.onrender.com`).
- When accessing the frontend URL, your browser will prompt for camera permissions. Click **Allow** and press **Start Camera**.

---

## 🔍 Verification & Health Checks

Once deployed, you can verify your deployment with these endpoints:

| Endpoint | Purpose | Expected Response |
| :--- | :--- | :--- |
| `GET /health` | Backend Liveness probe | `{"status": "healthy", "service": "CogniStream AI", ...}` |
| `GET /ready` | Backend Readiness probe | `{"status": "ready", "database": "connected", ...}` |
| `GET /docs` | Interactive Swagger API docs | OpenAPI UI |
| `GET /api/v1/roi/latest` | Latest face detections | `[...]` |

---

## 💡 Troubleshooting & Tips

### 1. Free Tier Cold Starts
- On Render's Free tier, the backend web service spins down after 15 minutes of inactivity.
- On first visit, the first request may take ~30-50 seconds while the container initializes.
- Once spun up, WebSockets and MJPEG streams run at full real-time speed.

### 2. WebSocket Connection
- The frontend automatically transforms `https://` into `wss://` for secure WebSocket streaming to `/api/v1/stream`.
- Make sure `VITE_API_URL` in the frontend static site is set to your backend URL (e.g. `https://cognistream-backend.onrender.com`).

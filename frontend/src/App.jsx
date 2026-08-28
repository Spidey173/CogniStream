import { useState, useEffect, useRef } from "react";
import "./App.css";
import "./VideoStreamer.css";
import VideoStreamer from "./VideoStreamer";

const RAW_API = import.meta.env.VITE_API_URL || "";
const API = RAW_API ? RAW_API.replace(/\/$/, "") : "";
const STREAM_URL = `${API}/api/v1/video`;
const ROI_URL = `${API}/api/v1/roi/latest`;
const POLL_INTERVAL = 300;

const formatNum = (num, decimals = 3) =>
  typeof num === "number" && !isNaN(num) ? num.toFixed(decimals) : "—";

const formatConf = (conf) =>
  typeof conf === "number" && !isNaN(conf) ? `${(conf * 100).toFixed(1)}%` : "—";

const EMOTION_EMOJIS = {
  Happy: "😊",
  Neutral: "😐",
  Surprise: "😲",
  Sad: "😔",
  Angry: "😠",
  Fear: "😨",
  Disgust: "😒",
  Contempt: "🤔",
};

const EMOTION_COLORS = {
  Happy: "#22c55e",
  Neutral: "#60a5fa",
  Surprise: "#f59e0b",
  Sad: "#94a3b8",
  Angry: "#ef4444",
  Fear: "#a855f7",
  Disgust: "#14b8a6",
  Contempt: "#ec4899",
};

const ALL_EMOTIONS = ["Happy", "Neutral", "Surprise", "Sad", "Angry", "Fear", "Disgust", "Contempt"];

function App() {
  const [roiData, setRoiData] = useState([]);
  const [liveAnalysis, setLiveAnalysis] = useState(null);
  const [streamOk, setStreamOk] = useState(false);
  const [error, setError] = useState(null);
  const [streamKey, setStreamKey] = useState(Date.now());
  const intervalRef = useRef(null);

  useEffect(() => {
    async function fetchROI() {
      try {
        const res = await fetch(`${ROI_URL}?count=8`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setRoiData(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err) {
        setError("Connecting to backend server (Render free instances take ~20s to wake up if sleeping)...");
      }
    }

    fetchROI();
    intervalRef.current = setInterval(fetchROI, POLL_INTERVAL);
    return () => clearInterval(intervalRef.current);
  }, []);

  const [liveFrame, setLiveFrame] = useState(null);

  const handleStatusChange = (status) => {
    if (status === "streaming") {
      setStreamKey(Date.now());
      setStreamOk(true);
      setError(null);
    } else if (status === "idle" || status === "error") {
      setStreamOk(false);
      setLiveAnalysis(null);
      setLiveFrame(null);
    }
  };

  const handleAnalysisUpdate = (detections, frameB64) => {
    if (Array.isArray(detections)) {
      if (detections.length > 0) {
        setLiveAnalysis(detections[0]);
      } else {
        setLiveAnalysis(null);
      }
    }
    if (frameB64) {
      setLiveFrame(`data:image/jpeg;base64,${frameB64}`);
      setStreamOk(true);
    }
  };

  const active = liveAnalysis;
  const currentEmotion = active?.emotion || (streamOk ? "No Face Detected" : "Camera Offline");
  const currentCondition = active?.condition || (streamOk ? "Position face in camera view" : "Click Start Camera to begin");
  const currentEmotionConf = active?.emotion_confidence || 0;
  const probabilities = active?.probabilities || {};

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-title-group">
          <h1>CogniStream AI</h1>
          <span className="subtitle">Real-Time Cognitive Face Condition & Emotion Analyzer</span>
        </div>
        <div className={`status-badge ${streamOk ? "" : "offline"}`}>
          <span className="status-dot" />
          {streamOk ? "AI Neural Engine Active" : "Waiting for Stream"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Webcam capture → WebSocket streamer */}
      <VideoStreamer
        fps={8}
        quality={0.6}
        width={480}
        height={360}
        onStatusChange={handleStatusChange}
        onAnalysisUpdate={handleAnalysisUpdate}
      />

      {/* Main content */}
      <div className="main-grid">
        {/* Video panel */}
        <div className="video-panel">
          <div className="video-panel-header">
            <span>📹</span> Live AI Annotated Stream (Face Detection & Emotion Neural Mesh)
          </div>
          <div className="video-container">
            {liveFrame ? (
              <img
                src={liveFrame}
                alt="Live AI Annotated Stream"
                className="live-ai-frame"
              />
            ) : (
              <img
                key={streamKey}
                src={`${STREAM_URL}?t=${streamKey}`}
                alt="Live face detection stream"
                crossOrigin="anonymous"
                onLoad={() => setStreamOk(true)}
                onError={() => {}}
              />
            )}
            {!streamOk && !liveFrame && (
              <div className="video-placeholder">
                <span>📡</span>
                Click "Start Camera" above to begin real-time face condition analysis
              </div>
            )}
          </div>
        </div>

        {/* Cognitive Condition & ROI Dashboard */}
        <aside className="roi-dashboard">
          {/* Main Face Condition Card */}
          <div className="roi-card condition-card">
            <div className="roi-card-title">🧠 Predicted Facial Condition & Mood</div>
            <div className="condition-hero">
              <div className="condition-emoji">
                {EMOTION_EMOJIS[currentEmotion] || "🧐"}
              </div>
              <div className="condition-text-group">
                <div className="condition-state">{currentCondition}</div>
                <div className="condition-sub">
                  Dominant State: <strong>{currentEmotion}</strong> ({currentEmotionConf.toFixed(1)}%)
                </div>
              </div>
            </div>
            {active?.track_id != null && (
              <div className="track-pill">Active Entity Track #{active.track_id}</div>
            )}
          </div>

          {/* 8-Class Emotion Spectrum Analysis */}
          <div className="roi-card">
            <div className="roi-card-title">📊 Real-Time Emotion Probability Distribution</div>
            <div className="spectrum-list">
              {ALL_EMOTIONS.map((em) => {
                const prob = probabilities[em] ?? 0;
                const isDominant = em === currentEmotion;
                return (
                  <div key={em} className={`spectrum-row ${isDominant ? "dominant" : ""}`}>
                    <div className="spectrum-label-group">
                      <span className="spectrum-emoji">{EMOTION_EMOJIS[em]}</span>
                      <span className="spectrum-name">{em}</span>
                    </div>
                    <div className="spectrum-bar-wrap">
                      <div
                        className="spectrum-bar-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, prob))}%`,
                          backgroundColor: EMOTION_COLORS[em] || "#6366f1",
                        }}
                      />
                    </div>
                    <span className="spectrum-val">{prob.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Real-Time Geometry & Kinematics */}
          <div className="roi-card">
            <div className="roi-card-title">📐 Live Face Position & Scale</div>
            {active ? (
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">X Center</span>
                  <span className="stat-value">{formatNum(active.x)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(0, active.x * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Y Center</span>
                  <span className="stat-value">{formatNum(active.y)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(0, active.y * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Face Width</span>
                  <span className="stat-value accent">{formatNum(active.width)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill accent"
                      style={{ width: `${Math.min(100, Math.max(0, active.width * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Face Height</span>
                  <span className="stat-value accent">{formatNum(active.height)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill accent"
                      style={{ width: `${Math.min(100, Math.max(0, active.height * 100))}%` }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">No face detected</div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;

import { useState, useEffect, useRef } from "react";
import "./App.css";
import "./VideoStreamer.css";
import VideoStreamer from "./VideoStreamer";

const RAW_API = import.meta.env.VITE_API_URL || "";
const API = RAW_API ? RAW_API.replace(/\/$/, "") : "";
const STREAM_URL = `${API}/api/v1/video`;
const ROI_URL = `${API}/api/v1/roi/latest`;
const POLL_INTERVAL = 250; // Ultra-responsive 4Hz live update

const formatNum = (num, decimals = 3) =>
  typeof num === "number" && !isNaN(num) ? num.toFixed(decimals) : "—";

const formatConf = (conf) =>
  typeof conf === "number" && !isNaN(conf) ? `${(conf * 100).toFixed(1)}%` : "—";

function App() {
  const [roiData, setRoiData] = useState([]);
  const [streamOk, setStreamOk] = useState(false);
  const [error, setError] = useState(null);
  const [streamKey, setStreamKey] = useState(Date.now());
  const intervalRef = useRef(null);

  useEffect(() => {
    async function fetchROI() {
      try {
        const res = await fetch(`${ROI_URL}?count=5`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setRoiData(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err) {
        setError("Cannot reach backend ROI server");
      }
    }

    fetchROI();
    intervalRef.current = setInterval(fetchROI, POLL_INTERVAL);
    return () => clearInterval(intervalRef.current);
  }, []);

  const handleStatusChange = (status) => {
    if (status === "streaming") {
      setStreamKey(Date.now());
      setStreamOk(true);
    } else if (status === "idle" || status === "error") {
      setStreamOk(false);
    }
  };

  const latest = roiData.length > 0 ? roiData[0] : null;

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>CogniStream AI</h1>
        <div className={`status-badge ${streamOk ? "" : "offline"}`}>
          <span className="status-dot" />
          {streamOk ? "Live AI Processing" : "Waiting"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Webcam capture → WebSocket streamer (optimized for cloud streaming) */}
      <VideoStreamer
        fps={8}
        quality={0.6}
        width={480}
        height={360}
        onStatusChange={handleStatusChange}
      />

      {/* Main content */}
      <div className="main-grid">
        {/* Video panel */}
        <div className="video-panel">
          <div className="video-panel-header">
            <span>📹</span> Live Processed Feed (MediaPipe AI Detection)
          </div>
          <div className="video-container">
            <img
              key={streamKey}
              src={`${STREAM_URL}?t=${streamKey}`}
              alt="Live face detection stream"
              crossOrigin="anonymous"
              onLoad={() => setStreamOk(true)}
              onError={() => setStreamOk(false)}
            />
            {!streamOk && (
              <div className="video-placeholder">
                <span>📡</span>
                Waiting for camera stream...
              </div>
            )}
          </div>
        </div>

        {/* ROI Dashboard */}
        <aside className="roi-dashboard">
          {/* Latest detection card with live visual position bars */}
          <div className="roi-card">
            <div className="roi-card-title">Live Face Coordinates</div>
            {latest ? (
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">X Position</span>
                  <span className="stat-value">{formatNum(latest.x)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(0, latest.x * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Y Position</span>
                  <span className="stat-value">{formatNum(latest.y)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(0, latest.y * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Face Width</span>
                  <span className="stat-value accent">{formatNum(latest.width)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill accent"
                      style={{ width: `${Math.min(100, Math.max(0, latest.width * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Face Height</span>
                  <span className="stat-value accent">{formatNum(latest.height)}</span>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill accent"
                      style={{ width: `${Math.min(100, Math.max(0, latest.height * 100))}%` }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">No detections yet</div>
            )}
          </div>

          {/* Confidence card with live gauge bar */}
          <div className="roi-card">
            <div className="roi-card-title">Detection Confidence</div>
            {latest ? (
              <div className="stat-item">
                <div className="confidence-row">
                  <span className="stat-value" style={{ fontSize: "1.6rem" }}>
                    {formatConf(latest.confidence)}
                  </span>
                  {latest.track_id != null && (
                    <span className="track-badge">ID #{latest.track_id}</span>
                  )}
                </div>
                <div className="metric-bar-bg" style={{ height: "8px", marginTop: "8px" }}>
                  <div
                    className="metric-bar-fill green"
                    style={{ width: `${Math.min(100, Math.max(0, (latest.confidence || 0) * 100))}%` }}
                  />
                </div>
              </div>
            ) : (
              <div className="empty-state">—</div>
            )}
          </div>

          {/* Recent detections list */}
          <div className="roi-card">
            <div className="roi-card-title">Real-Time Event Stream</div>
            {roiData.length > 0 ? (
              <ul className="roi-list">
                {roiData.map((roi, i) => (
                  <li key={roi.id ?? i} className="roi-list-item">
                    <span className="roi-coords">
                      ({formatNum(roi.x, 2)}, {formatNum(roi.y, 2)}) {formatNum(roi.width, 2)}×{formatNum(roi.height, 2)}
                    </span>
                    <span className="roi-time">
                      {roi.timestamp ? new Date(roi.timestamp).toLocaleTimeString() : ""}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty-state">Waiting for stream data...</div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;

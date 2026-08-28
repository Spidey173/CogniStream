import { useState, useEffect, useRef } from "react";
import "./App.css";
import "./VideoStreamer.css";
import VideoStreamer from "./VideoStreamer";

const API = import.meta.env.VITE_API_URL || "";
const STREAM_URL = `${API}/api/v1/video`;
const ROI_URL = `${API}/api/v1/roi/latest`;
const POLL_INTERVAL = 1000;

const formatNum = (num, decimals = 3) =>
  typeof num === "number" && !isNaN(num) ? num.toFixed(decimals) : "—";

const formatConf = (conf) =>
  typeof conf === "number" && !isNaN(conf) ? `${(conf * 100).toFixed(1)}%` : "—";

function App() {
  const [roiData, setRoiData] = useState([]);
  const [streamOk, setStreamOk] = useState(false);
  const [error, setError] = useState(null);
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

  const latest = roiData.length > 0 ? roiData[0] : null;

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>CogniStream AI</h1>
        <div className={`status-badge ${streamOk ? "" : "offline"}`}>
          <span className="status-dot" />
          {streamOk ? "Live" : "Waiting"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Webcam capture → WebSocket streamer */}
      <VideoStreamer />

      {/* Main content */}
      <div className="main-grid">
        {/* Video panel */}
        <div className="video-panel">
          <div className="video-panel-header">
            <span>📹</span> Live Feed
          </div>
          <div className="video-container">
            <img
              src={STREAM_URL}
              alt="Live face detection stream"
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
          {/* Latest detection card */}
          <div className="roi-card">
            <div className="roi-card-title">Latest Detection</div>
            {latest ? (
              <div className="stat-grid">
                <div className="stat-item">
                  <span className="stat-label">X</span>
                  <span className="stat-value">{formatNum(latest.x)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Y</span>
                  <span className="stat-value">{formatNum(latest.y)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Width</span>
                  <span className="stat-value accent">{formatNum(latest.width)}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Height</span>
                  <span className="stat-value accent">{formatNum(latest.height)}</span>
                </div>
              </div>
            ) : (
              <div className="empty-state">No detections yet</div>
            )}
          </div>

          {/* Confidence card */}
          <div className="roi-card">
            <div className="roi-card-title">Confidence</div>
            {latest ? (
              <div className="stat-item">
                <span className="stat-value" style={{ fontSize: "1.6rem" }}>
                  {formatConf(latest.confidence)}
                </span>
              </div>
            ) : (
              <div className="empty-state">—</div>
            )}
          </div>

          {/* Recent detections list */}
          <div className="roi-card">
            <div className="roi-card-title">Recent Detections</div>
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
              <div className="empty-state">Waiting for data...</div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;

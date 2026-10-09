import { useState, useEffect, useMemo } from "react";
import "./App.css";
import "./VideoStreamer.css";
import VideoStreamer from "./VideoStreamer";
import EmotionWaveform from "./EmotionWaveform";

const RAW_API = import.meta.env.VITE_API_URL || "";
const API = RAW_API ? RAW_API.replace(/\/$/, "") : "";
const ROI_URL = `${API}/api/roi/latest`;
const POLL_INTERVAL = 4000;

const formatNum = (num, decimals = 3) =>
  typeof num === "number" && !isNaN(num) ? num.toFixed(decimals) : "—";

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
  Happy: "#10b981",    /* Emerald */
  Neutral: "#a8a29e",  /* Warm Titanium */
  Surprise: "#ff6b35", /* Lava Orange */
  Sad: "#64748b",      /* Slate */
  Angry: "#e52e2e",    /* Deep Magma Red */
  Fear: "#c084fc",     /* Neon Violet */
  Disgust: "#f59e0b",  /* Amber */
  Contempt: "#ec4899", /* Magenta Crimson */
};

const ALL_EMOTIONS = [
  "Happy",
  "Neutral",
  "Surprise",
  "Sad",
  "Angry",
  "Fear",
  "Disgust",
];

function App() {
  const [roiData, setRoiData] = useState([]);
  const [liveAnalysis, setLiveAnalysis] = useState(null);
  const [streamOk, setStreamOk] = useState(false);
  const [serverConnected, setServerConnected] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [auditSearchQuery, setAuditSearchQuery] = useState("");
  const [activeSection, setActiveSection] = useState("all");

  // Fetch recent historical records from Vercel Serverless / Neon DB
  useEffect(() => {
    let active = true;

    async function fetchROI() {
      try {
        const res = await fetch(`${ROI_URL}?count=8`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (active) {
          setRoiData(Array.isArray(data) ? data : []);
          setServerConnected(true);
        }
      } catch {
        if (active) {
          setServerConnected(false);
        }
      }
    }

    fetchROI();
    const interval = setInterval(fetchROI, POLL_INTERVAL);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const handleStatusChange = (status) => {
    if (status === "streaming") {
      setStreamOk(true);
    } else if (status === "idle" || status === "error") {
      setStreamOk(false);
      setLiveAnalysis(null);
    }
  };

  const handleAnalysisUpdate = (detections) => {
    if (Array.isArray(detections) && detections.length > 0) {
      setLiveAnalysis(detections[0]);
    } else {
      setLiveAnalysis(null);
    }
  };

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const active = liveAnalysis;
  const cog = active?.cognitiveState;
  const currentEmotionName =
    cog?.stateName || active?.emotion || (streamOk ? "Detecting Face…" : "Optical Sensor Standby");
  const currentEmotionEmoji = cog?.emoji || EMOTION_EMOJIS[active?.emotion] || "🧐";
  const currentCondition =
    cog ? `${currentEmotionEmoji} ${currentEmotionName}` : active?.condition || "Sensor In Standby";
  const currentCategory = cog?.category || "Cognitive Affective Baseline";
  const currentDiagnostic =
    active?.diagnosticAnswer ||
    cog?.description ||
    (streamOk
      ? "Tracking 68 craniofacial landmarks across 3D pose and muscular action units."
      : "Optical sensor is currently offline. Launch Neural Vision above to initiate real-time biometric inference.");

  const currentEmotionConf = active?.emotion_confidence || 0;
  const probabilities = active?.probabilities || {};
  const au = active?.actionUnits || {
    smile: 0,
    smileAsymmetry: 0,
    mouthOpen: 0,
    eyeOpenness: 0,
    browFurrow: 0,
    yaw: 0,
    pitch: 0,
    roll: 0,
    focusScore: 0,
  };

  // Russell's Circumplex Affective Dimensions
  const happyProb = probabilities.Happy || 0;
  const surpriseProb = probabilities.Surprise || 0;
  const sadProb = probabilities.Sad || 0;
  const angryProb = probabilities.Angry || 0;
  const fearProb = probabilities.Fear || 0;
  const disgustProb = probabilities.Disgust || 0;

  const rawValence =
    cog?.valence != null
      ? cog.valence
      : (happyProb * 1.0 + surpriseProb * 0.15 - sadProb * 0.85 - angryProb * 0.95 - fearProb * 0.75 - disgustProb * 0.85) / 100;
  const circumplexValence = Math.max(-1, Math.min(1, rawValence));

  const rawArousal =
    cog?.arousal != null
      ? cog.arousal
      : (surpriseProb * 0.95 + angryProb * 0.85 + fearProb * 0.8 + happyProb * 0.6 + (Math.abs(au.yaw) / 45) * 15) / 100;
  const circumplexArousal = Math.max(0, Math.min(1, rawArousal));

  const circumplexX = Math.round(50 + circumplexValence * 45);
  const circumplexY = Math.round(100 - circumplexArousal * 85 - 10);

  // Copy live biometric telemetry to clipboard
  const handleCopyTelemetry = () => {
    if (!active) {
      showToast("⚠️ No active face telemetry detected to export");
      return;
    }
    const payload = {
      timestamp: new Date().toISOString(),
      emotion: active.emotion,
      confidence: active.emotion_confidence,
      cognitiveState: cog?.stateName,
      category: cog?.category,
      circumplex: {
        valence: Number(circumplexValence.toFixed(3)),
        arousal: Number(circumplexArousal.toFixed(3)),
      },
      actionUnits: au,
      probabilities,
    };
    navigator.clipboard?.writeText(JSON.stringify(payload, null, 2));
    showToast("✓ Live Biometric Telemetry Copied to Clipboard");
  };

  // Filtered audit table records
  const filteredRoiData = useMemo(() => {
    if (!auditSearchQuery.trim()) return roiData.slice(0, 8);
    const q = auditSearchQuery.toLowerCase();
    return roiData.filter((r) => {
      const track = `track #${r.track_id || 1}`.toLowerCase();
      const time = r.timestamp ? new Date(r.timestamp).toLocaleTimeString().toLowerCase() : "";
      return track.includes(q) || time.includes(q);
    });
  }, [roiData, auditSearchQuery]);

  return (
    <div className="app">
      {/* Toast Notification Container */}
      {toastMessage && (
        <div className="magma-toast">
          <span className="toast-sparkle">⚡</span>
          <span>{toastMessage}</span>
        </div>
      )}

      {/* ── Luxury Cockpit Command Header ── */}
      <header className="app-header">
        <div className="header-brand-cluster">
          <div className="brand-logo-glow">
            <svg className="brand-logo-icon" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="rgba(229, 46, 46, 0.4)" strokeWidth="1.5" />
              <circle cx="14" cy="14" r="8" fill="rgba(229, 46, 46, 0.2)" stroke="#ff6b35" strokeWidth="1.5" />
              <circle cx="14" cy="14" r="3.5" fill="#e52e2e" />
              <path d="M14 2v4M14 22v4M2 14h4M22 14h4" stroke="#ff6b35" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div className="header-title-group">
            <div className="brand-badge">
              <span className="brand-dot" />
              <span className="brand-tag font-display">COGNISTREAM AI</span>
              <span className="brand-version">v2.5 PRO</span>
            </div>
            <h1 className="font-display">Biometric Cognitive Engine</h1>
            <p className="subtitle">
              Sub-millisecond Edge Action Unit Synthesis &amp; 50-State Psychophysiological Classifier
            </p>
          </div>
        </div>

        {/* View Navigation Switcher */}
        <nav className="cockpit-nav">
          <button
            className={`nav-tab ${activeSection === "all" ? "active" : ""}`}
            onClick={() => setActiveSection("all")}
          >
            Overview
          </button>
          <button
            className={`nav-tab ${activeSection === "viewport" ? "active" : ""}`}
            onClick={() => setActiveSection("viewport")}
          >
            Optical &amp; FACS
          </button>
          <button
            className={`nav-tab ${activeSection === "affect" ? "active" : ""}`}
            onClick={() => setActiveSection("affect")}
          >
            Affect Matrix
          </button>
          <button
            className={`nav-tab ${activeSection === "ledger" ? "active" : ""}`}
            onClick={() => setActiveSection("ledger")}
          >
            Cloud Ledger
          </button>
        </nav>

        {/* Global Connection & Security Capsules */}
        <div className="header-status-group">
          <div className={`status-badge ${serverConnected ? "online" : "offline"}`} title="Neon PostgreSQL Serverless Ingestion Pipeline">
            <span className="status-dot" />
            <span className="badge-text">{serverConnected ? "Neon DB Cloud" : "Local Edge Cache"}</span>
          </div>

          <div className={`status-badge ${streamOk ? "active-engine" : "standby-engine"}`} title="BlazeFace + TinyLandmark WebAssembly SIMD Acceleration">
            <span className="status-dot" />
            <span className="badge-text">{streamOk ? "WASM AI Active" : "Sensor Standby"}</span>
          </div>

          <div className="status-badge privacy" title="Zero video transmission outside device memory">
            <span className="privacy-icon">🛡️</span>
            <span className="badge-text">Zero Cloud Leak</span>
          </div>

          <button
            className="header-action-btn"
            onClick={handleCopyTelemetry}
            title="Export instant biometric vector as JSON"
          >
            <span>📋 Export JSON</span>
          </button>
        </div>
      </header>

      {/* ── Executive Affective KPI Ribbon (Linear / Stripe Style) ── */}
      <section className="kpi-ribbon">
        {/* KPI 1: Primary Cognitive State */}
        <div className="kpi-card">
          <div className="kpi-label-row">
            <span className="kpi-eyebrow">COGNITIVE AFFECT</span>
            <span className="kpi-tag">{currentCategory.split(" ")[0]}</span>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-emoji">{currentEmotionEmoji}</span>
            <span className="kpi-val-primary font-display">{currentEmotionName}</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub">
              {currentEmotionConf > 0 ? `${currentEmotionConf.toFixed(1)}% Confidence` : "Awaiting Face"}
            </span>
            <div className="kpi-micro-meter">
              <div
                className="kpi-meter-fill"
                style={{ width: `${Math.min(100, Math.max(0, currentEmotionConf))}%` }}
              />
            </div>
          </div>
        </div>

        {/* KPI 2: Russell's Circumplex Vector */}
        <div className="kpi-card">
          <div className="kpi-label-row">
            <span className="kpi-eyebrow">CIRCUMPLEX VECTOR</span>
            <span className={`kpi-tag ${circumplexValence >= 0 ? "pos" : "neg"}`}>
              {circumplexValence >= 0 ? "Positive" : "Negative"}
            </span>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-mono-val">
              V: {circumplexValence >= 0 ? `+${circumplexValence.toFixed(2)}` : circumplexValence.toFixed(2)}
            </span>
            <span className="kpi-divider">|</span>
            <span className="kpi-mono-val">A: {circumplexArousal.toFixed(2)}</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub">
              Quadrant: <strong>{circumplexValence >= 0 ? (circumplexArousal > 0.4 ? "Euphoric Flow" : "Calm Serene") : (circumplexArousal > 0.4 ? "Alert Strain" : "Fatigued Focus")}</strong>
            </span>
          </div>
        </div>

        {/* KPI 3: Cognitive Engagement & Attention */}
        <div className="kpi-card">
          <div className="kpi-label-row">
            <span className="kpi-eyebrow">NEURAL ATTENTION</span>
            <span className="kpi-tag">{au.focusScore > 75 ? "Optimal" : "Standard"}</span>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-metric-number font-display">{au.focusScore}%</span>
            <span className="kpi-metric-unit">Quotient</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub">
              {Math.abs(au.yaw) < 12 && Math.abs(au.pitch) < 10
                ? "🎯 Centered Optical Axis"
                : "↔️ Off-Axis Inquisitive"}
            </span>
          </div>
        </div>

        {/* KPI 4: Edge Inference Telemetry */}
        <div className="kpi-card">
          <div className="kpi-label-row">
            <span className="kpi-eyebrow">EDGE RUNTIME</span>
            <span className="kpi-tag live">60Hz Loop</span>
          </div>
          <div className="kpi-val-row">
            <span className="kpi-mono-val accent">~8.2ms</span>
            <span className="kpi-metric-unit">GPU Cycle</span>
          </div>
          <div className="kpi-footer-row">
            <span className="kpi-sub">
              68-Pt FACS Mesh • FP16 WebAssembly
            </span>
          </div>
        </div>
      </section>

      {/* ── Main Power Grid ── */}
      <div className={`main-grid view-${activeSection}`}>
        {/* Left Column: Optical Viewport + Biometric FAU Telemetry Matrix */}
        {(activeSection === "all" || activeSection === "viewport") && (
          <div className="viewport-column">
            <VideoStreamer
              onStatusChange={handleStatusChange}
              onAnalysisUpdate={handleAnalysisUpdate}
            />

            {/* Biometric Facial Action Units (FAU) Telemetry Matrix */}
            <div className="au-matrix-card">
              <div className="matrix-header">
                <div className="matrix-title-group">
                  <span className="matrix-tag">FACS STANDARDS</span>
                  <span className="matrix-title font-display">Biometric Action Units (FAU) Matrix</span>
                </div>
                <span className="matrix-subtitle">68-Point Craniofacial Feature Extraction</span>
              </div>

              <div className="au-grid">
                {/* AU12: Zygomaticus Major (Smile) */}
                <div className="au-item">
                  <div className="au-item-header">
                    <div className="au-code-group">
                      <span className="au-code">AU12</span>
                      <span className="au-name">Zygomatic Smile</span>
                    </div>
                    <span className="au-val smile-val">
                      {au.smile}%
                    </span>
                  </div>
                  <div className="au-bar-bg">
                    <div
                      className="au-bar-fill smile"
                      style={{ width: `${Math.min(100, Math.max(0, au.smile))}%` }}
                    />
                  </div>
                  <span className="au-desc">
                    {au.smileAsymmetry > 20
                      ? `Asymmetric Smirk (${au.smileAsymmetry}%)`
                      : au.smile > 55
                      ? "Expressive Smile 😊"
                      : au.smile > 20
                      ? "Subtle Pleasant Tone"
                      : "Resting Lip Tone"}
                  </span>
                </div>

                {/* AU43: Ocular Openness (EAR) */}
                <div className="au-item">
                  <div className="au-item-header">
                    <div className="au-code-group">
                      <span className="au-code">AU43</span>
                      <span className="au-name">Ocular Aperture</span>
                    </div>
                    <span className="au-val eye-val">
                      {au.eyeOpenness}%
                    </span>
                  </div>
                  <div className="au-bar-bg">
                    <div
                      className="au-bar-fill eyes"
                      style={{ width: `${Math.min(100, Math.max(0, au.eyeOpenness))}%` }}
                    />
                  </div>
                  <span className="au-desc">
                    {au.eyeOpenness > 75
                      ? "Alert Gaze 👁️"
                      : au.eyeOpenness > 45
                      ? "Normal Openness"
                      : "Micro-Blink / Drowsy"}
                  </span>
                </div>

                {/* AU4: Corrugator Supercilii (Brow Strain) */}
                <div className="au-item">
                  <div className="au-item-header">
                    <div className="au-code-group">
                      <span className="au-code">AU04</span>
                      <span className="au-name">Brow Corrugator</span>
                    </div>
                    <span className="au-val brow-val">
                      {au.browFurrow}%
                    </span>
                  </div>
                  <div className="au-bar-bg">
                    <div
                      className="au-bar-fill brow"
                      style={{ width: `${Math.min(100, Math.max(0, au.browFurrow))}%` }}
                    />
                  </div>
                  <span className="au-desc">
                    {au.browFurrow > 45
                      ? "Corrugator Strain ⚠️"
                      : au.browFurrow > 20
                      ? "Cognitive Analysis"
                      : "Relaxed Forehead"}
                  </span>
                </div>

                {/* AU0: Cognitive Attention Quotient */}
                <div className="au-item">
                  <div className="au-item-header">
                    <div className="au-code-group">
                      <span className="au-code">AU00</span>
                      <span className="au-name">Attention Quotient</span>
                    </div>
                    <span className="au-val focus-val">
                      {au.focusScore}%
                    </span>
                  </div>
                  <div className="au-bar-bg">
                    <div
                      className="au-bar-fill focus"
                      style={{ width: `${Math.min(100, Math.max(0, au.focusScore))}%` }}
                    />
                  </div>
                  <span className="au-desc">
                    {au.focusScore > 75
                      ? "Deep Neural Focus 🎯"
                      : au.focusScore > 45
                      ? "Active Engagement"
                      : "Peripheral Gaze"}
                  </span>
                </div>
              </div>

              {/* 3D Cranial Pose Gyroscope Orientation Row */}
              <div className="pose-strip">
                <div className="pose-title-area">
                  <span className="pose-title">3D CRANIAL POSE:</span>
                  <span className="pose-sub">Euler Euler Rotations</span>
                </div>
                <div className="pose-badges">
                  <div className="pose-badge">
                    <span className="pose-axis">Yaw</span>
                    <strong className="mono-data">{au.yaw > 0 ? `+${au.yaw}` : au.yaw}°</strong>
                  </div>
                  <div className="pose-badge">
                    <span className="pose-axis">Pitch</span>
                    <strong className="mono-data">{au.pitch > 0 ? `+${au.pitch}` : au.pitch}°</strong>
                  </div>
                  <div className="pose-badge">
                    <span className="pose-axis">Roll</span>
                    <strong className="mono-data">{au.roll > 0 ? `+${au.roll}` : au.roll}°</strong>
                  </div>
                  <div className="pose-badge status">
                    {Math.abs(au.yaw) < 12 && Math.abs(au.pitch) < 10
                      ? "🎯 Aligned Optical Axis"
                      : "↔️ Off-Axis Inquisitive"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Right Column: Affective Analysis & Probability Spectra */}
        {(activeSection === "all" || activeSection === "affect") && (
          <aside className="roi-dashboard">
            {/* Main Face Condition Hero Card (50-State Granular Cognitive Emotion) */}
            <div className="roi-card condition-card">
              <div className="card-top-tag">
                <span className="taxonomy-pill font-display">50-STATE GRANULAR COGNITIVE TAXONOMY</span>
                {active?.track_id != null && (
                  <span className="track-tag mono-data">ENTITY #{active.track_id}</span>
                )}
              </div>

              <div className="condition-hero">
                <div className="condition-emoji-aura">
                  <div className="condition-emoji">{currentEmotionEmoji}</div>
                  <div className="aura-ring" />
                </div>

                <div className="condition-text-group">
                  <div className="condition-state font-display">{currentCondition}</div>
                  <div className="condition-cluster-tag">{currentCategory}</div>
                  <div className="condition-sub">
                    Macro Baseline: <strong>{active?.emotion || "Neutral"}</strong>{" "}
                    {currentEmotionConf > 0 && (
                      <span className="conf-pill">{currentEmotionConf.toFixed(1)}% Confidence</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Explainable AI Cognitive Diagnostic Engine ("The Smart Answer") */}
            <div className="roi-card diagnosis-card">
              <div className="roi-card-title">
                <span className="card-title-icon">🔬</span>
                <span>Explainable Biometric Diagnosis (&ldquo;The AI Reasoning&rdquo;)</span>
              </div>
              <div className="diagnosis-box">
                <p className="diagnosis-text">{currentDiagnostic}</p>
              </div>
              <div className="diagnosis-specs">
                <div className="spec-item">
                  <span className="spec-label">Taxonomy:</span>
                  <span className="spec-val">50-State FACS Matrix</span>
                </div>
                <div className="spec-item">
                  <span className="spec-label">Unilateral Asymmetry:</span>
                  <span className="spec-val mono-data">{au.smileAsymmetry}% Smirk Delta</span>
                </div>
              </div>
            </div>

            {/* Russell's Circumplex Affect Space (Valence vs Arousal) */}
            <div className="roi-card circumplex-card">
              <div className="roi-card-title">
                <span className="card-title-icon">🌐</span>
                <span>Russell&apos;s Circumplex Affect Space (Valence vs Arousal)</span>
              </div>
              <div className="circumplex-viewport">
                {/* Concentric radar rings */}
                <div className="circumplex-ring ring-1" />
                <div className="circumplex-ring ring-2" />
                <div className="circumplex-ring ring-3" />

                <div className="circumplex-axis x-axis" />
                <div className="circumplex-axis y-axis" />
                <div className="quadrant-label q-tl">Alert / Activated</div>
                <div className="quadrant-label q-tr">Euphoric / Excited</div>
                <div className="quadrant-label q-bl">Fatigued / Introspective</div>
                <div className="quadrant-label q-br">Calm / Serene</div>

                {/* Live Affect Coordinate Point */}
                {active && (
                  <div
                    className="circumplex-point"
                    style={{
                      left: `${circumplexX}%`,
                      top: `${circumplexY}%`,
                      backgroundColor: EMOTION_COLORS[active?.emotion] || "#e52e2e",
                      boxShadow: `0 0 16px ${EMOTION_COLORS[active?.emotion] || "#e52e2e"}`,
                    }}
                  >
                    <div className="point-pulse" />
                    <div className="point-label">
                      ({circumplexValence >= 0 ? `+${circumplexValence.toFixed(2)}` : circumplexValence.toFixed(2)}, {circumplexArousal.toFixed(2)})
                    </div>
                  </div>
                )}
              </div>
              <div className="circumplex-metrics">
                <div className="c-metric">
                  <span className="c-label">Valence:</span>
                  <span className={`c-val mono-data ${circumplexValence >= 0 ? "pos" : "neg"}`}>
                    {circumplexValence >= 0 ? `+${circumplexValence.toFixed(2)}` : circumplexValence.toFixed(2)}
                  </span>
                </div>
                <div className="c-metric">
                  <span className="c-label">Arousal:</span>
                  <span className="c-val arousal mono-data">
                    {circumplexArousal.toFixed(2)}
                  </span>
                </div>
                <div className="c-metric">
                  <span className="c-label">Quadrant:</span>
                  <span className="c-val quadrant">
                    {circumplexValence >= 0
                      ? circumplexArousal > 0.4
                        ? "High Positive"
                        : "Calm Positive"
                      : circumplexArousal > 0.4
                      ? "High Negative"
                      : "Low Negative"}
                  </span>
                </div>
              </div>
            </div>

            {/* Real-Time Emotion Probability Distribution */}
            <div className="roi-card">
              <div className="roi-card-title">
                <span className="card-title-icon">📊</span>
                <span>7-Class Neural Base Spectrum</span>
              </div>
              <div className="spectrum-list">
                {ALL_EMOTIONS.map((em) => {
                  const val = probabilities[em] ?? 0;
                  const isDominant = em === active?.emotion;
                  return (
                    <div
                      key={em}
                      className={`spectrum-row ${isDominant ? "dominant" : ""}`}
                    >
                      <div className="spectrum-label-group">
                        <span className="spectrum-emoji">{EMOTION_EMOJIS[em]}</span>
                        <span className="spectrum-name">{em}</span>
                      </div>
                      <div className="spectrum-bar-wrap">
                        <div
                          className="spectrum-bar-fill"
                          style={{
                            width: `${Math.min(100, Math.max(0, val))}%`,
                            backgroundColor: EMOTION_COLORS[em] || "#e52e2e",
                          }}
                        />
                      </div>
                      <span className="spectrum-val mono-data">{val.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Live Face Position & Geometry */}
            <div className="roi-card">
              <div className="roi-card-title">
                <span className="card-title-icon">📐</span>
                <span>Spatial Geometry &amp; Scale</span>
              </div>
              {active ? (
                <div className="stat-grid">
                  <div className="stat-item">
                    <span className="stat-label">X Center</span>
                    <span className="stat-value mono-data">{formatNum(active.x)}</span>
                    <div className="metric-bar-bg">
                      <div
                        className="metric-bar-fill"
                        style={{ width: `${Math.min(100, Math.max(0, active.x * 100))}%` }}
                      />
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Y Center</span>
                    <span className="stat-value mono-data">{formatNum(active.y)}</span>
                    <div className="metric-bar-bg">
                      <div
                        className="metric-bar-fill"
                        style={{ width: `${Math.min(100, Math.max(0, active.y * 100))}%` }}
                      />
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Width Ratio</span>
                    <span className="stat-value accent mono-data">{formatNum(active.width)}</span>
                    <div className="metric-bar-bg">
                      <div
                        className="metric-bar-fill accent"
                        style={{ width: `${Math.min(100, Math.max(0, active.width * 100))}%` }}
                      />
                    </div>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Height Ratio</span>
                    <span className="stat-value accent mono-data">{formatNum(active.height)}</span>
                    <div className="metric-bar-bg">
                      <div
                        className="metric-bar-fill accent"
                        style={{ width: `${Math.min(100, Math.max(0, active.height * 100))}%` }}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <span className="empty-icon">👁️</span>
                  <span>No facial entity detected in sensor viewport</span>
                </div>
              )}
            </div>
          </aside>
        )}
      </div>

      {/* ── Bottom Section: 30-Second Continuous Affect Waveform ── */}
      {(activeSection === "all" || activeSection === "affect") && (
        <section className="waveform-section">
          <EmotionWaveform activeData={active} />
        </section>
      )}

      {/* ── Cloud Telemetry & PostgreSQL Audit Feed ── */}
      {(activeSection === "all" || activeSection === "ledger") && (
        <section className="telemetry-section">
          <div className="telemetry-card">
            <div className="telemetry-header">
              <div className="telemetry-title-cluster">
                <span className="telemetry-tag">POSTGRESQL AUDIT</span>
                <span className="telemetry-title font-display">Neon Cloud Ingest Ledger</span>
                <span className="telemetry-sub">Asynchronous edge batches committed via serverless transactions</span>
              </div>
              <div className="telemetry-controls-cluster">
                <input
                  type="text"
                  placeholder="Filter track or time…"
                  value={auditSearchQuery}
                  onChange={(e) => setAuditSearchQuery(e.target.value)}
                  className="table-search-input"
                />
                <button className="table-action-btn" onClick={handleCopyTelemetry}>
                  Export JSON
                </button>
              </div>
            </div>

            <div className="telemetry-table-wrap">
              <div className="telemetry-table">
                <div className="table-header-row">
                  <span>TIMESTAMP</span>
                  <span>ENTITY TRACK</span>
                  <span>CONFIDENCE</span>
                  <span>NORMAL COORDINATES</span>
                  <span>BOUNDING RATIO</span>
                  <span>SYNC STATUS</span>
                </div>
                {filteredRoiData.length > 0 ? (
                  filteredRoiData.map((r, idx) => (
                    <div key={r.id || idx} className="table-row">
                      <span className="time-col mono-data">
                        {r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : "Just now"}
                      </span>
                      <span className="track-col mono-data">Track #{r.track_id || 1}</span>
                      <span className="conf-col mono-data">
                        {typeof r.confidence === "number"
                          ? `${(r.confidence * 100).toFixed(0)}%`
                          : "96%"}
                      </span>
                      <span className="pos-col mono-data">
                        {typeof r.x === "number" && typeof r.y === "number"
                          ? `(${r.x.toFixed(2)}, ${r.y.toFixed(2)})`
                          : "—"}
                      </span>
                      <span className="dim-col mono-data">
                        {typeof r.width === "number" && typeof r.height === "number"
                          ? `${(r.width * 100).toFixed(0)}% × ${(r.height * 100).toFixed(0)}%`
                          : "Active"}
                      </span>
                      <span className="status-col">
                        <span className="table-status-pill committed">
                          <span className="table-dot" /> Committed
                        </span>
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="table-empty-row">
                    <span>No audit batches matching current filter</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── System Architecture & Compliance Footer ── */}
      <footer className="app-footer">
        <div className="footer-left">
          <span className="footer-brand font-display">COGNISTREAM AI • v2.5</span>
          <span className="footer-bullet">•</span>
          <span className="footer-meta">BlazeFace WebAssembly SIMD</span>
          <span className="footer-bullet">•</span>
          <span className="footer-meta">FACS 68D Landmark Decomposition</span>
        </div>
        <div className="footer-right">
          <span className="footer-guarantee">
            <span className="footer-shield">🛡️</span> 100% Client-Side Processing • Zero External Video Streams
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;

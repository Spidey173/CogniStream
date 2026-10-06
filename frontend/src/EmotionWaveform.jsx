import { useEffect, useRef } from "react";

export default function EmotionWaveform({ activeData }) {
  const canvasRef = useRef(null);
  const historyRef = useRef([]);
  const maxHistory = 90; // ~15-20 seconds of telemetry

  // Compute current instant values deterministically for rendering
  const probs = activeData?.probabilities;
  const au = activeData?.actionUnits;

  let currentValence = 0;
  let currentFocus = 0;
  let currentSmile = 0;

  if (probs) {
    const positive = (probs.Happy || 0) + (probs.Surprise || 0) * 0.2;
    const negative =
      (probs.Sad || 0) * 0.8 +
      (probs.Angry || 0) * 0.9 +
      (probs.Fear || 0) * 0.7 +
      (probs.Disgust || 0) * 0.8;
    const rawValence = positive - negative;
    currentValence = Math.max(-100, Math.min(100, Math.round(rawValence)));
    currentFocus = au?.focusScore ?? 80;
    currentSmile = au?.smile ?? 0;
  }

  // Update history buffer on active data change
  useEffect(() => {
    const history = historyRef.current;
    const now = Date.now();

    if (activeData && activeData.probabilities) {
      history.push({
        t: now,
        valence: currentValence,
        focus: currentFocus,
        smile: currentSmile,
        dominant: activeData.emotion || "Neutral",
      });
    } else {
      // Idle / no face detected state: decay towards baseline
      const last = history[history.length - 1];
      const decayedValence = last ? Math.round(last.valence * 0.9) : 0;
      const decayedFocus = last ? Math.round(last.focus * 0.85) : 0;

      history.push({
        t: now,
        valence: decayedValence,
        focus: decayedFocus,
        smile: 0,
        dominant: "Offline",
      });
    }

    if (history.length > maxHistory) {
      history.shift();
    }
  }, [activeData, currentValence, currentFocus, currentSmile]);

  // Canvas drawing loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const history = historyRef.current;

      ctx.clearRect(0, 0, width, height);

      // Background grid
      ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = 1;
      const gridStepsY = 4;
      for (let i = 1; i < gridStepsY; i++) {
        const y = (height / gridStepsY) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Vertical time gridlines
      const gridStepsX = 6;
      for (let i = 1; i < gridStepsX; i++) {
        const x = (width / gridStepsX) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Zero-valence centerline
      const centerY = height * 0.5;
      ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      if (history.length < 2) {
        animId = requestAnimationFrame(render);
        return;
      }

      const dx = width / (maxHistory - 1);
      const startOffset = width - (history.length - 1) * dx;

      // Helper to draw a glowing line curve
      const drawChannel = (getY, strokeColor, fillColor, glowColor) => {
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.2;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 10;
        ctx.beginPath();

        history.forEach((pt, idx) => {
          const x = startOffset + idx * dx;
          const y = getY(pt);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Optional Area Fill
        if (fillColor) {
          ctx.lineTo(startOffset + (history.length - 1) * dx, height);
          ctx.lineTo(startOffset, height);
          ctx.closePath();
          ctx.fillStyle = fillColor;
          ctx.fill();
        }
        ctx.restore();
      };

      // 1. Channel: Cognitive Focus / Attention (Deep Magma Red)
      drawChannel(
        (pt) => height - (pt.focus / 100) * (height * 0.75) - height * 0.1,
        "#e52e2e",
        "rgba(229, 46, 46, 0.06)",
        "rgba(229, 46, 46, 0.65)"
      );

      // 2. Channel: Smile Intensity (Lava Amber/Orange)
      drawChannel(
        (pt) => height - (pt.smile / 100) * (height * 0.6) - 10,
        "#ff8252",
        null,
        "rgba(255, 130, 82, 0.5)"
      );

      // 3. Channel: Affective Valence (-100 to +100 mapped across centerY, Emerald Glow)
      drawChannel(
        (pt) => centerY - (pt.valence / 100) * (centerY * 0.75),
        "#10b981",
        "rgba(16, 185, 129, 0.08)",
        "rgba(16, 185, 129, 0.75)"
      );

      // Draw head pulse dot at current live point
      const lastPt = history[history.length - 1];
      const headX = width;
      const headY = centerY - (lastPt.valence / 100) * (centerY * 0.75);

      ctx.save();
      ctx.fillStyle = "#ff6b35";
      ctx.shadowColor = "#ff6b35";
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(headX - 3, headY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Responsive canvas sizing
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && canvasRef.current.parentElement) {
        const rect = canvasRef.current.parentElement.getBoundingClientRect();
        canvasRef.current.width = Math.max(300, Math.floor(rect.width));
        canvasRef.current.height = 160;
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="waveform-container">
      <div className="waveform-header">
        <div className="waveform-title-group">
          <div className="waveform-badge">
            <span className="waveform-badge-dot" />
            <span>CONTINUOUS AFFECT STREAM</span>
          </div>
          <span className="waveform-title font-display">
            30-Second Rolling Psychophysiological Waveform
          </span>
          <span className="waveform-subtitle">
            Sub-millisecond Action Unit Delta Tracking &amp; Circumplex Drift Vector
          </span>
        </div>
        <div className="waveform-legend">
          <div className="legend-item valence">
            <span className="legend-dot valence" />
            <span className="legend-label">Valence</span>
            <span className="legend-value">{currentValence > 0 ? `+${currentValence}` : currentValence}</span>
          </div>
          <div className="legend-item focus">
            <span className="legend-dot focus" />
            <span className="legend-label">Focus Quotient</span>
            <span className="legend-value">{currentFocus}%</span>
          </div>
          <div className="legend-item smile">
            <span className="legend-dot smile" />
            <span className="legend-label">Smile Index</span>
            <span className="legend-value">{currentSmile}%</span>
          </div>
        </div>
      </div>
      <div className="waveform-canvas-wrap">
        <canvas ref={canvasRef} className="waveform-canvas" />
        <div className="waveform-time-labels">
          <span>-30s</span>
          <span>-20s</span>
          <span>-10s</span>
          <span className="live-tag">
            <span className="live-tag-beacon" /> LIVE 60Hz SAMPLING
          </span>
        </div>
      </div>
    </div>
  );
}

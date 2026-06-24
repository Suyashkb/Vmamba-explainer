import { useEffect } from "react";
import Equation from "../shared/Equation";
import useScanAnimation from "../../hooks/useScanAnimation";
import toyTrace from "../../data/toy_trace.json";

/**
 * L3 — SSM Cell live trace.
 *
 * Driven entirely by toy_trace.json (4×4 toy image, d_state=16).
 * The step scrubber (or play button) animates through t=0…15:
 *
 *  • StateBarChart — 16 bars showing h_t filling/decaying
 *  • Strip heatmaps — Δ_t, B_t (16), C_t (16), Ā_t (16)
 *  • Live equation — \textcolor highlights the value changing at t
 *  • Spatial mini-grid — shows which token in the 4×4 scan is active
 *
 * Phase 3 goal: scrubbing must match toy_trace numerically (allclose 1e-4).
 * Validation: toy_trace was generated & validated in generate_toy_trace.py.
 */

const { tokens, d_state, H, W } = toyTrace;
const L = tokens.length;  // 16

// ---- Colour helpers --------------------------------------------------------
// Map a signed float to a red–white–blue diverging ramp
function divColor(v, vabs) {
  const t = Math.max(0, Math.min(1, v / (vabs || 1) * 0.5 + 0.5));
  const r = t < 0.5 ? 220 : Math.round(220 - (t - 0.5) * 2 * 170);
  const b = t > 0.5 ? 220 : Math.round(220 - (0.5 - t) * 2 * 170);
  const g = Math.round(30 + Math.min(t, 1 - t) * 2 * 90);
  return `rgb(${r},${g},${b})`;
}

// ---- StateBarChart ---------------------------------------------------------
function StateBarChart({ h, prev }) {
  const maxAbs = Math.max(...h.map(Math.abs), 0.01);
  return (
    <div className="state-bar-chart" aria-label="SSM hidden state h_t">
      {h.map((v, i) => {
        const pct     = Math.abs(v / maxAbs) * 100;
        const positive = v >= 0;
        const changed  = prev && Math.abs(v - prev[i]) > 1e-5;
        return (
          <div key={i} className="bar-wrapper"
            title={`h[${i}] = ${v.toFixed(5)}`}>
            <div className="bar-track">
              <div
                className={`bar-fill ${positive ? "pos" : "neg"} ${changed ? "changed" : ""}`}
                style={{
                  height: `${pct}%`,
                  bottom: positive ? "50%" : undefined,
                  top:    positive ? undefined : "50%",
                  transition: "height 0.12s ease",
                }}
              />
            </div>
            <span className="bar-label">{i}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---- Strip heatmap (1-D array → colour row) --------------------------------
function StripHeatmap({ values, label, colorFn }) {
  if (!values?.length) return null;
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);
  return (
    <div className="strip-heatmap">
      <span className="strip-label">{label}</span>
      <div className="strip-cells">
        {values.map((v, i) => (
          <div key={i} className="strip-cell"
            title={`[${i}] = ${v.toFixed(4)}`}
            style={{ background: colorFn(v, maxAbs) }}
          />
        ))}
      </div>
    </div>
  );
}

// ---- Live equation with textcolor highlighting ----------------------------
function LiveEquation({ tok, prev }) {
  // Values that changed compared to previous step get highlighted
  const xFmt   = tok.x.toFixed(3);
  const dFmt   = tok.delta.toFixed(3);
  const yFmt   = tok.y.toFixed(3);

  // KaTeX \textcolor works in mathmode
  const eqH = `h_t = \\bar{A}_t \\odot h_{t-1} + \\bar{B}_t \\odot \\textcolor{#f4a442}{x_t}`;
  const eqY = `y_t = C_t \\cdot h_t + D \\cdot \\textcolor{#f4a442}{x_t}`;

  return (
    <div className="live-equation">
      <Equation tex={eqH} display />
      <Equation tex={eqY} display />
      <div className="live-vals">
        <span>
          <Equation tex={`x_t = \\textcolor{#f4a442}{${xFmt}}`} />
        </span>
        <span>
          <Equation tex={`\\Delta_t = \\textcolor{#54c97f}{${dFmt}}`} />
        </span>
        <span>
          <Equation tex={`y_t = \\textcolor{#e05d78}{${yFmt}}`} />
        </span>
      </div>
    </div>
  );
}

// ---- Spatial mini-grid (4×4) -----------------------------------------------
function SpatialContext({ step, onSeek }) {
  return (
    <section className="grid-context">
      <h4>Spatial position — token {step} of {L - 1}</h4>
      <svg width={H * 30} height={W * 30} aria-label="4×4 scan grid">
        {tokens.map((_, i) => {
          const r = Math.floor(i / W);
          const c = i % W;
          const isCurrent = i === step;
          const isPast    = i < step;
          return (
            <g key={i} transform={`translate(${c * 30},${r * 30})`}
              style={{ cursor: "pointer" }}
              onClick={() => onSeek(i)}
              role="button"
              aria-label={`Jump to t=${i}`}>
              <rect width={28} height={28} rx={4}
                fill={isCurrent ? "var(--color-accent)"
                  : isPast ? "var(--color-cell-visited)"
                           : "var(--color-cell-bg)"}
                stroke="var(--color-cell-border)" strokeWidth={0.5}
              />
              <text x={14} y={16} textAnchor="middle"
                fontSize="9" fill="var(--color-text-secondary)"
                fontFamily="var(--font-mono)">
                {i}
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}

// ---- Root component --------------------------------------------------------
function L3_SSMCell() {
  const { step, playing, play, pause, seek, reset } =
    useScanAnimation(L, { fps: 4 }); // slow for clarity

  // Reset to step 0 on mount
  useEffect(() => { reset(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const tok     = tokens[step];
  const prevTok = step > 0 ? tokens[step - 1] : null;

  // Build Δ as a single-element array for the strip (it's a scalar here)
  const deltaArr = [tok.delta];

  return (
    <div className="level-container l3">
      <header className="level-header">
        <span className="level-badge">L3</span>
        <h2>SSM Cell — State Trace</h2>
        <p>4×4 toy scan, d_state={d_state}. Scrub or play to watch h_t evolve.</p>
      </header>

      {/* Step scrubber */}
      <div className="step-scrubber">
        <button onClick={() => seek(Math.max(0, step - 1))}
          disabled={step === 0} aria-label="Previous">‹</button>
        <input type="range" min={0} max={L - 1} value={step}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label={`Step ${step} of ${L - 1}`}
        />
        <button onClick={() => seek(Math.min(L - 1, step + 1))}
          disabled={step === L - 1} aria-label="Next">›</button>
        <button className="scan-btn" onClick={playing ? pause : play}
          style={{ marginLeft: 4 }}>
          {playing ? "⏸" : "▶"}
        </button>
        <span className="step-label">t = {step}</span>
      </div>

      {/* Live update equation */}
      <LiveEquation tok={tok} prev={prevTok} />

      {/* Hidden state bar chart */}
      <section className="state-section">
        <h4>Hidden state h<sub>t</sub> — {d_state} dimensions</h4>
        <StateBarChart h={tok.h} prev={prevTok?.h} />
      </section>

      {/* Parameter strips */}
      <section className="strips-section">
        <h4>Per-token parameters</h4>
        <StripHeatmap values={deltaArr} label="Δ"
          colorFn={(v, m) => {
            const a = 0.15 + (v / m) * 0.85;
            return `rgba(244,164,66,${a})`;
          }}
        />
        <StripHeatmap values={tok.B} label="B"
          colorFn={(v, m) => divColor(v, m)} />
        <StripHeatmap values={tok.C} label="C"
          colorFn={(v, m) => divColor(v, m)} />
        <StripHeatmap values={tok.Abar} label="Ā"
          colorFn={(v, m) => {
            const a = 0.15 + (v / m) * 0.85;
            return `rgba(224,93,120,${a})`;
          }}
        />
      </section>

      {/* Spatial position */}
      <SpatialContext step={step} onSeek={seek} />
    </div>
  );
}

export default L3_SSMCell;

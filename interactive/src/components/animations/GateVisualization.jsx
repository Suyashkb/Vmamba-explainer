import { useState } from "react";
import { HeatmapCells } from "../shared/Heatmap";
import blockData from "../../data/activations/stage2_block0.json";

/**
 * GateVisualization — gate_heatmap as an opacity mask over the feature map.
 *
 * Three panels:
 *   [feat pre-gate]  ⊙ SiLU  [gate values]  =  [feat × gate]
 *
 * A slider blends the first panel from pre-gate to post-gate, making it
 * viscerally clear how the gate suppresses low-activation spatial regions.
 */

const { feat_heatmap, gate_heatmap } = blockData;

// Compute post-gate heatmap once at module level (no hooks needed)
const postGateValues = feat_heatmap.values.map((v, i) => v * gate_heatmap.values[i]);
const postGate = {
  h: feat_heatmap.h,
  w: feat_heatmap.w,
  values: postGateValues,
  vmin: Math.min(...postGateValues),
  vmax: Math.max(...postGateValues),
};

const SZ = 76;   // heatmap thumbnail size (px)

// ---- Labelled heatmap panel ------------------------------------------------
function HeatPanel({ data, label, color }) {
  return (
    <div className="gate-panel">
      <span className="gate-panel-label" style={{ color }}>{label}</span>
      <svg width={SZ} height={SZ} style={{ display: "block" }}>
        <HeatmapCells data={data} width={SZ} height={SZ} />
      </svg>
    </div>
  );
}

// ---- Blended pre/post view (slider-driven) ---------------------------------
function BlendedPanel({ alpha }) {
  const { h, w } = feat_heatmap;
  return (
    <div className="gate-panel">
      <span className="gate-panel-label" style={{ color: "#4e9af1" }}>
        {alpha < 0.05 ? "Pre-gate z" : alpha > 0.95 ? "Post-gate z⊙g" : "Blended"}
      </span>
      <svg width={SZ} height={SZ} style={{ display: "block" }}>
        {/* Base: feat heatmap */}
        <HeatmapCells data={feat_heatmap} width={SZ} height={SZ} />
        {/* Dark overlay per cell proportional to (1 - gate) × alpha */}
        {gate_heatmap.values.map((gv, i) => {
          const row = Math.floor(i / w);
          const col = i % w;
          const cw  = SZ / w;
          const ch  = SZ / h;
          const darkOpacity = alpha * (1 - gv) * 0.82;
          return darkOpacity > 0.01 ? (
            <rect key={i}
              x={col * cw} y={row * ch} width={cw} height={ch}
              fill="black" opacity={darkOpacity}
            />
          ) : null;
        })}
      </svg>
    </div>
  );
}

// ---- Main component --------------------------------------------------------
function GateVisualization() {
  const [alpha, setAlpha] = useState(0);

  return (
    <div className="gate-viz">
      {/* Three-panel equation */}
      <div className="gate-panels-row">
        <BlendedPanel alpha={alpha} />

        <div className="gate-op-col">
          <span className="gate-op-sym">⊙</span>
          <span className="gate-op-sub">SiLU</span>
        </div>

        <HeatPanel data={gate_heatmap} label="Gate g" color="#54c97f" />

        <div className="gate-op-col">
          <span className="gate-op-sym">=</span>
        </div>

        <HeatPanel data={postGate} label="z⊙SiLU(g)" color="#f4a442" />
      </div>

      {/* Gate strength slider */}
      <div className="gate-slider-row">
        <span className="gate-slider-label" style={{ color: "#4e9af1" }}>Pre</span>
        <input
          type="range" min={0} max={100} value={Math.round(alpha * 100)}
          onChange={(e) => setAlpha(e.target.value / 100)}
          aria-label="Gate blend strength"
          className="gate-slider"
          style={{ accentColor: "#54c97f" }}
        />
        <span className="gate-slider-label" style={{ color: "#f4a442" }}>Post</span>
      </div>
      <p className="gate-note">
        Slide right — dark cells are gated to near-zero.
        Gate values (green) near 0 block; near 1 pass through.
      </p>
    </div>
  );
}

export default GateVisualization;

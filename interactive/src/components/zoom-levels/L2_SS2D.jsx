import { useState } from "react";
import ScanSweep from "../animations/ScanSweep";
import { HeatmapCells } from "../shared/Heatmap";
import useScanAnimation from "../../hooks/useScanAnimation";
import useZoomLevel from "../../hooks/useZoomLevel";
import blockData from "../../data/activations/stage2_block0.json";

/**
 * L2 — SS2D Cross-Scan  (the "money shot").
 *
 * Phase 4 upgrade: main grid rendered by ScanSweep (canvas).
 *   - Zero React re-renders per animation tick
 *   - One ImageData blit + O(trail) ctx.fillRect calls per frame
 *
 * Still shows: direction legend, play controls, mini-heatmap merge row.
 */

const { feat_heatmap, gate_heatmap, scan_paths, H, W } = blockData;
const L = H * W;  // 196

const DIR_COLORS = ["#4e9af1", "#f4a442", "#54c97f", "#e05d78"];
const DIR_LABELS = ["→ Row L→R", "↓ Col T→B", "← Row R→L", "↑ Col B→T"];
const PERMS      = [0, 1, 2, 3].map((d) => scan_paths[`d${d}`]);
const MINI_SZ    = 76;
const CELL       = 32;

// ---- Play controls ---------------------------------------------------------
function PlayControls({ playing, play, pause, reset, step }) {
  return (
    <div className="scan-controls">
      <button className="scan-btn" onClick={playing ? pause : play}
        aria-label={playing ? "Pause" : "Play"}>
        {playing ? "⏸" : "▶"}
      </button>
      <button className="scan-btn" onClick={reset} aria-label="Reset">⏮</button>
      <div className="scan-progress"
        role="progressbar" aria-valuenow={step} aria-valuemin={0} aria-valuemax={L - 1}>
        <div className="scan-progress-fill"
          style={{ width: `${(step / (L - 1)) * 100}%` }} />
      </div>
      <span className="scan-step-label">{step} / {L - 1}</span>
    </div>
  );
}

// ---- Direction filter legend -----------------------------------------------
function DirectionLegend({ filterDir, onToggle }) {
  return (
    <div className="dir-legend">
      {DIR_LABELS.map((label, i) => (
        <button key={i}
          className={`dir-chip ${filterDir === null || filterDir === i ? "active" : "dim"}`}
          onClick={() => onToggle(i)}>
          <span className="dir-chip-dot" style={{ background: DIR_COLORS[i] }} />
          {label}
        </button>
      ))}
      <button className="dir-chip active" onClick={() => onToggle(null)}>All</button>
    </div>
  );
}

// ---- Mini heatmaps showing per-direction scan progress --------------------
function MiniHeatmapRow({ step }) {
  return (
    <div className="mini-heatmap-row">
      {[0, 1, 2, 3].map((d) => {
        const visited = new Set(PERMS[d].slice(0, step + 1));
        return (
          <div key={d} className="mini-heatmap-card">
            <span className="mini-label" style={{ color: DIR_COLORS[d] }}>
              {DIR_LABELS[d]}
            </span>
            <svg width={MINI_SZ} height={MINI_SZ} style={{ display: "block" }}>
              <HeatmapCells data={feat_heatmap} width={MINI_SZ} height={MINI_SZ} />
              {/* Direction tint */}
              <rect width={MINI_SZ} height={MINI_SZ}
                fill={DIR_COLORS[d]} opacity={0.15} />
              {/* Visited cells */}
              {[...visited].map((idx) => {
                const sr = Math.floor(idx / W);
                const sc = idx % W;
                const cw = MINI_SZ / W;
                const ch = MINI_SZ / H;
                return (
                  <rect key={idx}
                    x={sc * cw} y={sr * ch} width={cw - 0.5} height={ch - 0.5}
                    fill="white" opacity={0.18}
                  />
                );
              })}
            </svg>
          </div>
        );
      })}

      <div className="merge-arrow" aria-hidden>⊕</div>

      <div className="mini-heatmap-card">
        <span className="mini-label" style={{ color: "var(--color-text-secondary)" }}>
          Merged
        </span>
        <svg width={MINI_SZ} height={MINI_SZ} style={{ display: "block" }}>
          <HeatmapCells data={feat_heatmap} width={MINI_SZ} height={MINI_SZ} />
        </svg>
      </div>
    </div>
  );
}

// ---- Root ------------------------------------------------------------------
function L2_SS2D() {
  const { zoomIn }    = useZoomLevel();
  const { step, playing, play, pause, reset } =
    useScanAnimation(L, { fps: 14, loop: true });

  const [filterDir, setFilterDir] = useState(null);
  const activeDirs = filterDir === null
    ? new Set([0, 1, 2, 3])
    : new Set([filterDir]);

  const handleFilterToggle = (d) =>
    setFilterDir((prev) => (prev === d ? null : d));

  return (
    <div className="level-container l2">
      <header className="level-header">
        <span className="level-badge">L2</span>
        <h2>SS2D — Cross-Scan</h2>
        <p>
          Canvas-rendered grid — zero React re-renders per frame.{" "}
          <button className="zoom-into-btn inline"
            onClick={() => zoomIn("ssm_cell")}>
            Zoom into SSM Cell →
          </button>
        </p>
      </header>

      <PlayControls playing={playing} play={play} pause={pause}
        reset={reset} step={step} />
      <DirectionLegend filterDir={filterDir} onToggle={handleFilterToggle} />

      <div className="l2-body">
        <div className="scan-grid-wrap">
          {/* Canvas-based ScanSweep — Phase 4 performance upgrade */}
          <ScanSweep
            step={step}
            activeDirs={activeDirs}
            heatmapData={feat_heatmap}
            scanPaths={scan_paths}
            H={H}
            W={W}
            cellSize={CELL}
          />
        </div>
      </div>

      <MiniHeatmapRow step={step} />
    </div>
  );
}

export default L2_SS2D;

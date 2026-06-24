import { useRef, useEffect, useMemo } from "react";
import { viridisRGB } from "../shared/Heatmap";
import useReducedMotion from "../../hooks/useReducedMotion";

/**
 * ScanSweep — canvas-based scan-cursor visualisation.
 *
 * Performance contract (Phase-4 gotcha):
 *   Never setState per frame. All per-frame work is canvas draw calls:
 *   one putImageData (base heatmap) + O(TRAIL_LEN × 4) fillRect calls.
 *   Zero React reconciliation per animation tick.
 *
 * @param {number}   step        current scan timestep (0..L-1)
 * @param {Set}      activeDirs  which direction cursors to show
 * @param {object}   heatmapData { h, w, values, vmin, vmax }
 * @param {object}   scanPaths   { d0, d1, d2, d3 } — permutation arrays
 * @param {number}   H           grid height (cells)
 * @param {number}   W           grid width  (cells)
 * @param {number}   [cellSize=32]
 */

const TRAIL_LEN = 10;

// Direction RGBA components (matching DIR_COLORS in L2)
const DIR_RGBA = [
  [78,  154, 241],   // #4e9af1  blue
  [244, 164,  66],   // #f4a442  orange
  [84,  201, 127],   // #54c97f  green
  [224,  93, 120],   // #e05d78  red
];

function ScanSweep({
  step,
  activeDirs,
  heatmapData,
  scanPaths,
  H,
  W,
  cellSize = 32,
}) {
  const canvasRef = useRef(null);
  const reducedMotion = useReducedMotion();

  // ---- One-time precompute: viridis ImageData for the base heatmap --------
  const baseImageData = useMemo(() => {
    const { values, vmin, vmax } = heatmapData;
    const range = (vmax - vmin) || 1;
    const pw = W * cellSize;
    const ph = H * cellSize;
    const img = new ImageData(pw, ph);
    const d = img.data;

    for (let gy = 0; gy < H; gy++) {
      for (let gx = 0; gx < W; gx++) {
        const idx = gy * W + gx;
        const t = (values[idx] - vmin) / range;
        const [r, g, b] = viridisRGB(t);

        // Fill (cellSize-1) × (cellSize-1) pixels; leave 1-px gap for grid
        for (let py = 0; py < cellSize - 1; py++) {
          for (let px = 0; px < cellSize - 1; px++) {
            const pi = ((gy * cellSize + py) * pw + (gx * cellSize + px)) * 4;
            d[pi]     = r;
            d[pi + 1] = g;
            d[pi + 2] = b;
            d[pi + 3] = 255;
          }
        }
      }
    }
    return img;
  }, [heatmapData, H, W, cellSize]);

  // ---- Static fallback for reduced-motion ----------------------------------
  useEffect(() => {
    if (!reducedMotion) return;
    const canvas = canvasRef.current;
    if (!canvas || !baseImageData) return;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(baseImageData, 0, 0);
    // Draw all scan paths at once with low opacity trails
    const scanPathsArr = [
      scanPaths.d0, scanPaths.d1, scanPaths.d2, scanPaths.d3,
    ];
    scanPathsArr.forEach((path, d) => {
      const [r, g, b] = DIR_RGBA[d];
      const color = `rgb(${r},${g},${b})`;
      ctx.globalAlpha = 0.18;
      path.forEach((cellIdx) => {
        const row = Math.floor(cellIdx / W);
        const col = cellIdx % W;
        ctx.fillStyle = color;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
      });
    });
    ctx.globalAlpha = 1.0;
  }, [reducedMotion, baseImageData, scanPaths, H, W, cellSize]);

  // ---- Per-frame: draw base + overlays ------------------------------------
  useEffect(() => {
    if (reducedMotion) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Base heatmap — single blit, no loops
    ctx.putImageData(baseImageData, 0, 0);

    const perms = [
      scanPaths.d0, scanPaths.d1, scanPaths.d2, scanPaths.d3,
    ];

    for (let d = 0; d < 4; d++) {
      if (!activeDirs.has(d)) continue;
      const [r, g, b] = DIR_RGBA[d];
      const perm = perms[d];

      // Trail — fading semi-transparent rects
      const trailStart = Math.max(0, step - TRAIL_LEN);
      for (let s = trailStart; s < step; s++) {
        const si = perm[s];
        if (si == null) continue;
        const row = Math.floor(si / W);
        const col = si % W;
        const age = step - s;                           // 1..TRAIL_LEN
        const alpha = (1 - age / (TRAIL_LEN + 1)) * 0.38;
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
      }

      // Cursor — current position
      const ci = perm[step];
      if (ci == null) continue;
      const crow = Math.floor(ci / W);
      const ccol = ci % W;
      ctx.fillStyle = `rgba(${r},${g},${b},0.55)`;
      ctx.fillRect(ccol * cellSize, crow * cellSize, cellSize - 1, cellSize - 1);
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(
        ccol * cellSize + 1,
        crow * cellSize + 1,
        cellSize - 3,
        cellSize - 3,
      );

      // Step label inside cursor cell
      ctx.fillStyle = "white";
      ctx.font = `bold 9px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        String(step),
        ccol * cellSize + cellSize / 2,
        crow * cellSize + cellSize / 2,
      );
    }
  }, [reducedMotion, step, activeDirs, baseImageData, scanPaths, H, W, cellSize]);

  return (
    <canvas
      ref={canvasRef}
      width={W * cellSize}
      height={H * cellSize}
      className="scan-sweep-canvas"
      aria-label="SS2D scan grid"
    />
  );
}

export default ScanSweep;

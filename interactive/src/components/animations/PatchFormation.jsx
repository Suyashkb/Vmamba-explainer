import { useRef, useEffect, useCallback, useState } from "react";
import useReducedMotion from "../../hooks/useReducedMotion";

/**
 * PatchFormation — canvas animation of image → patch grid → token embeddings.
 *
 * Three-phase animation (each ~50 frames at 30fps):
 *   Phase 0 (0-40):   Synthetic HSL image fades in
 *   Phase 1 (40-100): 14×14 patch-grid lines appear progressively
 *   Phase 2 (100-180): Bezier projection rays shoot from each patch
 *                       centre to a token dot below (L→R, T→B order)
 *   Phase 3 (180-220): Token grid fills; rays fade out
 *
 * Uses ImageData → canvas; all animation is rAF-driven with a frame counter.
 * Zero React state updates per frame.
 */

// ---- Layout ----------------------------------------------------------------
const CW = 288;          // canvas width
const CH = 260;          // canvas height
const IMG_SIZE = 180;    // image square side (px)
const IMG_X = (CW - IMG_SIZE) / 2;
const IMG_Y = 10;
const PATCHES = 7;       // 7×7 = 49 patches (visually clear, not 14×14)
const PATCH_PX = IMG_SIZE / PATCHES;  // pixels per patch
const TOKEN_Y = IMG_Y + IMG_SIZE + 24; // y of token row
const TOKEN_R = 4;       // token dot radius
const TOKEN_SPACING = CW / (PATCHES * PATCHES + 1); // ~1.6px — will use grid
// Layout: tokens arranged 14 per row, 4 rows below image
const TOKENS_PER_ROW = Math.ceil(PATCHES * PATCHES / 4);

const TOTAL_FRAMES = 220;

// ---- Synthetic image -------------------------------------------------------
// HSL colour wheel — x maps to hue, y maps to saturation
function buildImageData() {
  const img = new ImageData(IMG_SIZE, IMG_SIZE);
  const d = img.data;
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      // Hue from angle + some concentric ring modulation
      const cx = x - IMG_SIZE / 2;
      const cy = y - IMG_SIZE / 2;
      const angle = Math.atan2(cy, cx) / (2 * Math.PI) + 0.5;
      const dist  = Math.sqrt(cx * cx + cy * cy) / (IMG_SIZE / 2);
      const hue   = (angle * 360 + dist * 120) % 360;
      const sat   = 0.6 + 0.4 * Math.sin(dist * Math.PI);
      const lum   = 0.35 + 0.25 * Math.cos(dist * Math.PI * 3);

      // HSL → RGB
      const [r, g, b] = hslToRgb(hue, sat, lum);
      const pi = (y * IMG_SIZE + x) * 4;
      d[pi]     = r;
      d[pi + 1] = g;
      d[pi + 2] = b;
      d[pi + 3] = 255;
    }
  }
  return img;
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if      (h < 60)  { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else              { r = c; g = 0; b = x; }
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

// Precompute image once
const IMAGE_DATA = buildImageData();

// Patch centre positions (in canvas coords)
const PATCH_CENTERS = [];
for (let py = 0; py < PATCHES; py++) {
  for (let px = 0; px < PATCHES; px++) {
    PATCH_CENTERS.push({
      x: IMG_X + px * PATCH_PX + PATCH_PX / 2,
      y: IMG_Y + py * PATCH_PX + PATCH_PX / 2,
    });
  }
}
const N_PATCHES = PATCHES * PATCHES;  // 49

// Token positions: 2 rows of 25/24 dots centred below image
const TOKEN_POSITIONS = PATCH_CENTERS.map((_, i) => {
  const row = Math.floor(i / TOKENS_PER_ROW);
  const col = i % TOKENS_PER_ROW;
  const rowY = TOKEN_Y + row * 14;
  const rowCount = row < Math.floor(N_PATCHES / TOKENS_PER_ROW)
    ? TOKENS_PER_ROW
    : N_PATCHES % TOKENS_PER_ROW;
  const totalW = (rowCount - 1) * (CW / TOKENS_PER_ROW);
  const startX = (CW - totalW) / 2;
  return { x: startX + col * (CW / TOKENS_PER_ROW), y: rowY };
});

// ---- Drawing helpers -------------------------------------------------------
function drawFrame(ctx, frame) {
  ctx.clearRect(0, 0, CW, CH);

  // ---- Phase 0: Image fade-in (0-40) ----------------------------------------
  const imgAlpha = Math.min(1, frame / 40);
  if (imgAlpha > 0) {
    ctx.save();
    ctx.globalAlpha = imgAlpha;
    // Use offscreen canvas to blit ImageData
    const offscreen = document.createElement("canvas");
    offscreen.width  = IMG_SIZE;
    offscreen.height = IMG_SIZE;
    offscreen.getContext("2d").putImageData(IMAGE_DATA, 0, 0);
    ctx.drawImage(offscreen, IMG_X, IMG_Y);
    ctx.restore();
  }

  // ---- Phase 1: Grid lines (40-100) -----------------------------------------
  if (frame >= 40) {
    const gridProgress = Math.min(1, (frame - 40) / 60);
    const linesDrawn   = Math.floor(gridProgress * (PATCHES - 1) * 2); // H + V lines

    ctx.save();
    ctx.strokeStyle = "rgba(200,211,240,0.7)";
    ctx.lineWidth   = 0.8;

    // Horizontal lines
    for (let i = 1; i < PATCHES && (i - 1) * 2 < linesDrawn; i++) {
      const ly = IMG_Y + i * PATCH_PX;
      ctx.beginPath();
      ctx.moveTo(IMG_X, ly);
      ctx.lineTo(IMG_X + IMG_SIZE, ly);
      ctx.stroke();
    }
    // Vertical lines
    for (let i = 1; i < PATCHES && (i - 1 + (PATCHES - 1)) * 2 < linesDrawn * 2; i++) {
      const lx = IMG_X + i * PATCH_PX;
      ctx.beginPath();
      ctx.moveTo(lx, IMG_Y);
      ctx.lineTo(lx, IMG_Y + IMG_SIZE);
      ctx.stroke();
    }

    // Image border
    ctx.strokeStyle = "rgba(78,154,241,0.8)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(IMG_X, IMG_Y, IMG_SIZE, IMG_SIZE);
    ctx.restore();
  }

  // ---- Phase 2: Projection rays + token dots (100-180) ----------------------
  if (frame >= 100) {
    const rayProgress  = Math.min(1, (frame - 100) / 80);
    const raysComplete = Math.floor(rayProgress * N_PATCHES);

    for (let i = 0; i < raysComplete; i++) {
      const pc = PATCH_CENTERS[i];
      const tc = TOKEN_POSITIONS[i];
      const rayAlpha = frame < 180 ? 0.5 : Math.max(0, 1 - (frame - 180) / 40);

      // Bezier ray
      ctx.save();
      ctx.strokeStyle = `rgba(84,201,127,${rayAlpha * 0.6})`;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(pc.x, pc.y);
      ctx.bezierCurveTo(pc.x, pc.y + 30, tc.x, tc.y - 20, tc.x, tc.y);
      ctx.stroke();
      ctx.restore();

      // Token dot
      ctx.save();
      ctx.fillStyle = `rgba(78,154,241,${0.3 + rayAlpha * 0.7})`;
      ctx.beginPath();
      ctx.arc(tc.x, tc.y, TOKEN_R, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Patch highlight overlay for the current patch being projected
    if (raysComplete < N_PATCHES) {
      const pi  = raysComplete;
      const row = Math.floor(pi / PATCHES);
      const col = pi % PATCHES;
      ctx.save();
      ctx.fillStyle = "rgba(84,201,127,0.3)";
      ctx.fillRect(
        IMG_X + col * PATCH_PX,
        IMG_Y + row * PATCH_PX,
        PATCH_PX,
        PATCH_PX,
      );
      ctx.restore();
    }
  }

  // ---- Labels ---------------------------------------------------------------
  ctx.save();
  ctx.fillStyle = "rgba(122,138,176,0.9)";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  if (frame >= 40)  ctx.fillText("224×224 → 7×7 patches", CW / 2, IMG_Y - 2);
  if (frame >= 160) ctx.fillText("49 token embeddings", CW / 2, TOKEN_Y + 30);
  ctx.restore();
}

// ---- Component -------------------------------------------------------------
function PatchFormation({ compact = false }) {
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);
  const frameRef  = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [done, setDone]       = useState(false);
  const reducedMotion = useReducedMotion();

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setPlaying(false);
  }, []);

  const startLoop = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (reducedMotion) {
      // Skip to final frame immediately
      frameRef.current = 220;
      drawFrame(ctx, frameRef.current);
      setPlaying(false);
      setDone(true);
      return;
    }

    const tick = () => {
      frameRef.current += 1;
      drawFrame(ctx, frameRef.current);
      if (frameRef.current < TOTAL_FRAMES) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setPlaying(false);
        setDone(true);
        rafRef.current = null;
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    setPlaying(true);
    setDone(false);
  }, [reducedMotion]);

  const handlePlay = () => {
    if (playing) { stop(); return; }
    if (done) { frameRef.current = 0; }
    startLoop();
  };

  const handleReset = () => {
    stop();
    frameRef.current = 0;
    setDone(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, CW, CH);
  };

  // Draw initial frame on mount
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawFrame(ctx, 0);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  return (
    <div className="patch-formation">
      <canvas
        ref={canvasRef}
        width={CW}
        height={CH}
        className="patch-canvas"
        aria-label="Patch formation animation"
      />
      <div className="anim-controls">
        <button className="scan-btn" onClick={handlePlay}
          aria-label={playing ? "Pause" : done ? "Replay" : "Play"}>
          {playing ? "⏸" : done ? "↺" : "▶"}
        </button>
        <button className="scan-btn" onClick={handleReset} aria-label="Reset">⏮</button>
        <span className="scan-step-label">Patch embed animation</span>
      </div>
    </div>
  );
}

export default PatchFormation;

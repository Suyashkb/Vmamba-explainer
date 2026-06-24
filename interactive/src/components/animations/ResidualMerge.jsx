import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * ResidualMerge — two token streams converging at ⊕.
 *
 * Streams:
 *   z″  (out_proj output, SS2D path) — enters from the left
 *   x   (skip connection, x_in)      — enters from below
 *
 * Play animates:
 *   1. Left stream traces its path  (0.8s)
 *   2. Skip stream traces its path  (0.8s, starts 0.3s in)
 *   3. ⊕ symbol pulses             (0.6s, starts at 1.0s)
 *   4. Output stream traces right   (0.6s, starts at 1.3s)
 *
 * Uses Framer Motion pathLength animation on <motion.path>.
 */

// ---- SVG layout ------------------------------------------------------------
const W = 288;
const H = 180;
const CX = W / 2;    // ⊕ centre x
const CY = H / 2;    // ⊕ centre y
const R  = 16;       // ⊕ circle radius

// Path descriptors
const PATH_LEFT  = `M 8,${CY} L ${CX - R - 4},${CY}`;
const PATH_SKIP  = `M ${CX},${H - 8} L ${CX},${CY + R + 4}`;
const PATH_OUT   = `M ${CX + R + 4},${CY} L ${W - 8},${CY}`;

// ---- Animated path ---------------------------------------------------------
function AnimPath({ d, color, duration, delay, strokeW = 2 }) {
  return (
    <motion.path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={strokeW}
      strokeLinecap="round"
      initial={{ pathLength: 0, opacity: 0.9 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={{ duration, delay, ease: "easeInOut" }}
    />
  );
}

// ---- Arrowhead at end of path ----------------------------------------------
function Arrow({ x, y, dir, color }) {
  // dir: "right" | "up"
  const pts = dir === "right"
    ? `${x},${y - 5} ${x + 9},${y} ${x},${y + 5}`
    : `${x - 5},${y} ${x},${y - 9} ${x + 5},${y}`;
  return (
    <motion.polygon
      points={pts}
      fill={color}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: dir === "right" ? 1.9 : dir === "up" ? 1.1 : 0.8, duration: 0.2 }}
    />
  );
}

// ---- ⊕ circle --------------------------------------------------------------
function PlusCircle({ pulse }) {
  return (
    <motion.g
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: pulse ? [1, 1.25, 1] : 1, opacity: 1 }}
      transition={{ delay: 1.0, duration: pulse ? 0.5 : 0.3, ease: "easeOut" }}
      style={{ originX: `${CX}px`, originY: `${CY}px` }}
    >
      <circle cx={CX} cy={CY} r={R} fill="var(--color-surface-2)"
        stroke="#c8d3f0" strokeWidth={1.5} />
      <text x={CX} y={CY + 5} textAnchor="middle"
        fill="#c8d3f0" fontSize={18} fontFamily="monospace">⊕</text>
    </motion.g>
  );
}

// ---- Stream labels ---------------------------------------------------------
function StreamLabel({ x, y, text, color, delay }) {
  return (
    <motion.text
      x={x} y={y}
      fill={color} fontSize={10} fontFamily="monospace"
      textAnchor="middle"
      initial={{ opacity: 0 }}
      animate={{ opacity: 0.8 }}
      transition={{ delay, duration: 0.3 }}
    >
      {text}
    </motion.text>
  );
}

// ---- Main component --------------------------------------------------------
function ResidualMerge() {
  const [key, setKey] = useState(0);   // remount to replay

  return (
    <div className="residual-merge">
      <AnimatePresence mode="wait">
        <motion.svg
          key={key}
          width={W} height={H}
          className="residual-svg"
          aria-label="Residual merge animation"
        >
          {/* ---- z″ stream (left) ---- */}
          <AnimPath d={PATH_LEFT} color="#4e9af1" duration={0.8} delay={0} strokeW={2.5} />
          <Arrow x={CX - R - 4} y={CY} dir="right" color="#4e9af1" />
          <StreamLabel x={50} y={CY - 10} text="z″ (SS2D output)" color="#4e9af1" delay={0.1} />

          {/* ---- skip stream (bottom) ---- */}
          <AnimPath d={PATH_SKIP} color="#54c97f" duration={0.8} delay={0.3} strokeW={2.5} />
          <Arrow x={CX} y={CY + R + 4} dir="up" color="#54c97f" />
          <StreamLabel x={CX + 38} y={H - 14} text="x_in (skip)" color="#54c97f" delay={0.4} />

          {/* ---- ⊕ node ---- */}
          <PlusCircle pulse />

          {/* ---- output stream (right) ---- */}
          <AnimPath d={PATH_OUT} color="#f4a442" duration={0.6} delay={1.3} strokeW={2.5} />
          <Arrow x={W - 8} y={CY} dir="right" color="#f4a442" />
          <StreamLabel x={W - 36} y={CY - 10} text="x_out" color="#f4a442" delay={1.5} />

          {/* Equation */}
          <motion.text x={CX} y={H - 8} textAnchor="middle"
            fill="var(--color-text-muted)" fontSize={10} fontFamily="monospace"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ delay: 1.8 }}>
            x_out = x_in + z″
          </motion.text>
        </motion.svg>
      </AnimatePresence>

      <div className="anim-controls">
        <button className="scan-btn" onClick={() => setKey((k) => k + 1)}
          aria-label="Replay">↺ replay</button>
        <span className="scan-step-label">Residual connection</span>
      </div>
    </div>
  );
}

export default ResidualMerge;

import { useRef, useState } from "react";
import useZoomLevel from "../../hooks/useZoomLevel";
import useExplainerStore from "../../store/explainerStore";

/**
 * L1 — VSS Block, hand-laid SVG.
 *
 * Nodes are React <g> elements with fixed x/y; edges are SVG <path>s.
 * Token flow is animated with SMIL <animateMotion> + <mpath> — native SVG
 * animation that requires no per-frame React state.
 *
 * Play/Pause uses SVGSVGElement.pauseAnimations() / unpauseAnimations().
 * Clicking the SS2D node navigates to L2.
 */

// ---- Layout ----------------------------------------------------------------
const VB_W = 1100;
const VB_H = 420;

const Y_MAIN  = 198;   // center-y of main-path nodes
const Y_UPPER = 115;   // center-y of upper branch (DWConv → SS2D)
const Y_LOWER = 281;   // center-y of lower branch (gate)

const NODE_H  = 44;
const NODE_R  = 6;     // border-radius

const NODES = [
  { id: "ln",       label: "LayerNorm",     sub: null,    x:   50, cy: Y_MAIN,  w: 110 },
  { id: "in_proj",  label: "in_proj ×2",   sub: null,    x:  225, cy: Y_MAIN,  w: 110 },
  { id: "dw_conv",  label: "DW Conv",       sub: null,    x:  405, cy: Y_UPPER, w: 110 },
  { id: "ss2d",     label: "SS2D",          sub: "→ L2",  x:  580, cy: Y_UPPER, w: 110, zoomable: true },
  { id: "gate",     label: "gate SiLU",     sub: null,    x:  405, cy: Y_LOWER, w: 110 },
  { id: "mul_op",   label: "⊙",             sub: null,    x:  745, cy: Y_MAIN,  w:  60 },
  { id: "out_proj", label: "out_proj",      sub: null,    x:  860, cy: Y_MAIN,  w: 110 },
  { id: "add_op",   label: "⊕",             sub: "skip",  x: 1000, cy: Y_MAIN,  w:  60 },
];

// Precompute left/right edges of each node (for edge attachment)
const nodeMap = Object.fromEntries(
  NODES.map((n) => [n.id, { ...n, left: n.x, right: n.x + n.w, top: n.cy - NODE_H / 2 }])
);

const N = nodeMap;

// Edge path descriptors: id, d (SVG path), dotColor, dotDelay, dotDur
const EDGES = [
  { id: "e_in_ln",        d: `M0,${Y_MAIN} L${N.ln.left},${Y_MAIN}`,                                               color: "#4e9af1", delay: 0.0,  dur: 0.8 },
  { id: "e_ln_ip",        d: `M${N.ln.right},${Y_MAIN} L${N.in_proj.left},${Y_MAIN}`,                              color: "#4e9af1", delay: 0.3,  dur: 0.9 },
  { id: "e_ip_dw",        d: `M${N.in_proj.right},${Y_MAIN} C${N.in_proj.right+35},${Y_MAIN} ${N.dw_conv.left-35},${Y_UPPER} ${N.dw_conv.left},${Y_UPPER}`, color: "#4e9af1", delay: 0.6,  dur: 1.1 },
  { id: "e_ip_gate",      d: `M${N.in_proj.right},${Y_MAIN} C${N.in_proj.right+35},${Y_MAIN} ${N.gate.left-35},${Y_LOWER} ${N.gate.left},${Y_LOWER}`,       color: "#54c97f", delay: 0.6,  dur: 1.1 },
  { id: "e_dw_ss2d",      d: `M${N.dw_conv.right},${Y_UPPER} L${N.ss2d.left},${Y_UPPER}`,                          color: "#4e9af1", delay: 1.0,  dur: 0.9 },
  { id: "e_ss2d_mul",     d: `M${N.ss2d.right},${Y_UPPER} C${N.ss2d.right+28},${Y_UPPER} ${N.mul_op.left-28},${Y_MAIN} ${N.mul_op.left},${Y_MAIN}`,         color: "#4e9af1", delay: 1.4,  dur: 1.0 },
  { id: "e_gate_mul",     d: `M${N.gate.right},${Y_LOWER} C${N.gate.right+115},${Y_LOWER} ${N.mul_op.left-28},${Y_MAIN} ${N.mul_op.left},${Y_MAIN}`,          color: "#54c97f", delay: 1.0,  dur: 1.3 },
  { id: "e_mul_op",       d: `M${N.mul_op.right},${Y_MAIN} L${N.out_proj.left},${Y_MAIN}`,                          color: "#f4a442", delay: 1.7,  dur: 0.8 },
  { id: "e_op_add",       d: `M${N.out_proj.right},${Y_MAIN} L${N.add_op.left},${Y_MAIN}`,                          color: "#f4a442", delay: 2.1,  dur: 0.8 },
  { id: "e_add_out",      d: `M${N.add_op.right},${Y_MAIN} L${VB_W},${Y_MAIN}`,                                     color: "#c8d3f0", delay: 2.5,  dur: 0.6 },
  // skip residual: x_in → bottom curve → ⊕
  { id: "e_skip",         d: `M0,${Y_MAIN} C0,380 ${N.add_op.left + N.add_op.w / 2},380 ${N.add_op.left + N.add_op.w / 2},${N.add_op.top + NODE_H}`, color: "#6b7fa0", delay: 0.0, dur: 2.8 },
];

// ---- Sub-components --------------------------------------------------------

function NodeBox({ node, isSelected, onClick }) {
  const { id, label, sub, x, cy, w, zoomable } = node;
  const y = cy - NODE_H / 2;

  const fill   = isSelected ? "var(--color-node-selected-bg)"
               : zoomable   ? "var(--color-node-zoomable-bg)"
               : "var(--color-node-bg)";
  const stroke = isSelected ? "var(--color-node-selected-border)"
               : zoomable   ? "var(--color-accent)"
               : "var(--color-node-border)";

  return (
    <g
      transform={`translate(${x},${y})`}
      onClick={() => onClick(node)}
      style={{ cursor: zoomable ? "pointer" : "default" }}
      role={zoomable ? "button" : undefined}
    >
      <rect width={w} height={NODE_H} rx={NODE_R}
        fill={fill} stroke={stroke} strokeWidth={zoomable || isSelected ? 1.5 : 1}
      />
      <text x={w / 2} y={sub ? 16 : NODE_H / 2 + 1}
        textAnchor="middle" dominantBaseline="middle"
        fill={zoomable ? "var(--color-accent)" : "var(--color-node-text)"}
        fontSize="12" fontFamily="var(--font-mono)"
        fontWeight={zoomable ? "600" : "400"}
      >
        {label}
      </text>
      {sub && (
        <text x={w / 2} y={NODE_H - 10}
          textAnchor="middle"
          fill={id === "add_op" ? "var(--color-shape-text)" : "var(--color-accent)"}
          fontSize="9" fontFamily="var(--font-mono)">
          {sub}
        </text>
      )}
    </g>
  );
}

// ---- Main component --------------------------------------------------------

function L1_VSSBlock() {
  const svgRef = useRef(null);
  const [flowOn, setFlowOn] = useState(true);

  const { zoomIn } = useZoomLevel();
  const setSelectedNode = useExplainerStore((s) => s.setSelectedNode);
  const selectedNode    = useExplainerStore((s) => s.selectedNode);

  const toggleFlow = () => {
    if (!svgRef.current) return;
    if (flowOn) {
      svgRef.current.pauseAnimations();
      setFlowOn(false);
    } else {
      svgRef.current.unpauseAnimations();
      setFlowOn(true);
    }
  };

  const handleNodeClick = (node) => {
    setSelectedNode(node.id);
    if (node.zoomable) {
      setTimeout(() => zoomIn(node.id), 110);
    }
  };

  return (
    <div className="level-container l1">
      <header className="level-header">
        <span className="level-badge">L1</span>
        <h2>VSS Block</h2>
        <p>
          Click <strong>SS2D</strong> to zoom into the cross-scan.
          <button className="flow-toggle-btn" onClick={toggleFlow}>
            {flowOn ? "⏸ pause flow" : "▶ play flow"}
          </button>
        </p>
      </header>

      <div className="l1-svg-wrap">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          className="l1-svg"
        >
          <defs>
            {/* Arrow marker */}
            <marker id="l1-arrow" viewBox="0 -4 8 8" refX="7" refY="0"
                    markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,-4L8,0L0,4" fill="var(--color-edge)" />
            </marker>

            {/* Branch labels */}
            <style>{`
              .l1-edge-label {
                font: 10px var(--font-mono);
                fill: var(--color-text-muted);
              }
            `}</style>
          </defs>

          {/* Branch zone backgrounds */}
          <rect x={395} y={70} width={320} height={68} rx={4}
            fill="#1c2b4a22" stroke="#4e9af133" strokeWidth={1}
          />
          <rect x={395} y={236} width={120} height={68} rx={4}
            fill="#54c97f11" stroke="#54c97f33" strokeWidth={1}
          />

          {/* Branch labels */}
          <text x={398} y={67} className="l1-edge-label">z (SSM input)</text>
          <text x={398} y={233} className="l1-edge-label">g (gate)</text>

          {/* x_in / x_out labels */}
          <text x={3} y={Y_MAIN - 12}
            fill="var(--color-text-secondary)" fontSize="11"
            fontFamily="var(--font-mono)">x_in</text>
          <text x={VB_W - 38} y={Y_MAIN - 12}
            fill="var(--color-text-secondary)" fontSize="11"
            fontFamily="var(--font-mono)">x_out</text>

          {/* Edges with arrowheads */}
          {EDGES.map((e) => (
            <path key={e.id} id={e.id} d={e.d}
              fill="none" stroke="var(--color-edge)" strokeWidth={1.5}
              markerEnd={e.id !== "e_skip" ? "url(#l1-arrow)" : undefined}
            />
          ))}

          {/* Skip arrow (enters from bottom of ⊕) */}
          <line
            x1={N.add_op.left + N.add_op.w / 2}
            y1={N.add_op.top + NODE_H}
            x2={N.add_op.left + N.add_op.w / 2}
            y2={N.add_op.top + NODE_H - 1}
            stroke="var(--color-edge)" strokeWidth={1.5}
            markerEnd="url(#l1-arrow)"
          />

          {/* Animated flow dots (SMIL animateMotion) */}
          {EDGES.map((e) => (
            <circle key={`dot-${e.id}`} r={4} fill={e.color} opacity={0.92}>
              <animateMotion
                dur={`${e.dur}s`}
                repeatCount="indefinite"
                begin={`${e.delay}s`}
              >
                <mpath href={`#${e.id}`} />
              </animateMotion>
            </circle>
          ))}

          {/* Nodes (drawn on top of edges) */}
          {NODES.map((node) => (
            <NodeBox
              key={node.id}
              node={node}
              isSelected={node.id === selectedNode}
              onClick={handleNodeClick}
            />
          ))}
        </svg>
      </div>
    </div>
  );
}

export default L1_VSSBlock;

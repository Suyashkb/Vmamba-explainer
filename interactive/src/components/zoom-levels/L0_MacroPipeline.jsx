import { useMemo } from "react";
import { HeatmapCells } from "../shared/Heatmap";
import useZoomLevel from "../../hooks/useZoomLevel";
import useExplainerStore from "../../store/explainerStore";

// Stage feature heatmap data (one representative block per stage)
import s0 from "../../data/activations/stage0_block0.json";
import s1 from "../../data/activations/stage1_block0.json";
import s2 from "../../data/activations/stage2_block0.json";
import s3 from "../../data/activations/stage3_block0.json";

const STAGE_ACT = [s0, s1, s2, s3];

// ---- Layout constants -------------------------------------------------------
const SVG_W = 1120;
const SVG_H = 300;

// Op nodes (no heatmap)
const OP_W = 90;
const OP_H = 52;
// Stage nodes (with heatmap thumbnail)
const ST_W = 110;
const ST_H = 130;
const HM_SZ = 68; // heatmap thumbnail size

const Y_OP   = (SVG_H - OP_H) / 2;          // 124
const Y_ST   = (SVG_H - ST_H) / 2;          // 85

// Pipeline items (left → right)
const ITEMS = [
  { id: "patch_embed", label: "Patch\nEmbed",  type: "op",    shape: "(B,56²,96)" },
  { id: "stage0",      label: "Stage 0",       type: "stage", stageIdx: 0, depth: "2×VSS", zoomTo: "stage0_block0" },
  { id: "down0",       label: "↓ 2×",          type: "op",    shape: "(B,28²,192)" },
  { id: "stage1",      label: "Stage 1",       type: "stage", stageIdx: 1, depth: "2×VSS", zoomTo: "stage1_block0" },
  { id: "down1",       label: "↓ 2×",          type: "op",    shape: "(B,14²,384)" },
  { id: "stage2",      label: "Stage 2",       type: "stage", stageIdx: 2, depth: "9×VSS", zoomTo: "stage2_block0" },
  { id: "down2",       label: "↓ 2×",          type: "op",    shape: "(B,7²,768)" },
  { id: "stage3",      label: "Stage 3",       type: "stage", stageIdx: 3, depth: "2×VSS", zoomTo: "stage3_block0" },
  { id: "norm",        label: "LN",            type: "op",    shape: "(B,768)" },
  { id: "head",        label: "Head\n1000",    type: "op",    shape: "(B,1000)" },
];

// Compute x positions so the pipeline fits the SVG width with even spacing
const GAPS = ITEMS.length - 1;
const totalContentW = ITEMS.reduce((acc, item) => acc + (item.type === "stage" ? ST_W : OP_W), 0);
const spacing = (SVG_W - totalContentW - 40) / GAPS;   // ~16 px gap

const NODES = (() => {
  let cx = 20;
  return ITEMS.map((item) => {
    const w = item.type === "stage" ? ST_W : OP_W;
    const h = item.type === "stage" ? ST_H : OP_H;
    const y = item.type === "stage" ? Y_ST : Y_OP;
    const node = { ...item, x: cx, y, w, h, cx: cx + w / 2, cy: SVG_H / 2 };
    cx += w + spacing;
    return node;
  });
})();

const nodeById = Object.fromEntries(NODES.map((n) => [n.id, n]));

// Edges: simple lines connecting adjacent items
const EDGES = NODES.slice(0, -1).map((src, i) => {
  const tgt = NODES[i + 1];
  return {
    x1: src.x + src.w,
    y1: SVG_H / 2,
    x2: tgt.x,
    y2: SVG_H / 2,
  };
});

// ---- Component --------------------------------------------------------------
function L0_MacroPipeline() {
  const { zoomIn } = useZoomLevel();
  const setSelectedNode = useExplainerStore((s) => s.setSelectedNode);
  const selectedNode    = useExplainerStore((s) => s.selectedNode);

  const handleClick = (node) => {
    setSelectedNode(node.id);
    if (node.zoomTo) {
      setTimeout(() => zoomIn(node.zoomTo), 100);
    }
  };

  return (
    <div className="level-container l0">
      <header className="level-header">
        <span className="level-badge">L0</span>
        <h2>VMamba-Tiny Architecture</h2>
        <p>Click a Stage node to zoom into one VSS block.</p>
      </header>

      <div className="l0-svg-wrap">
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="l0-svg">
          <defs>
            <marker id="l0-arrow" viewBox="0 -4 8 8" refX="7" refY="0"
                    markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,-4L8,0L0,4" fill="var(--color-edge)" />
            </marker>
          </defs>

          {/* Edges */}
          {EDGES.map((e, i) => (
            <line key={i}
              x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke="var(--color-edge)" strokeWidth={1.5}
              markerEnd="url(#l0-arrow)"
            />
          ))}

          {/* Nodes */}
          {NODES.map((node) => {
            const isSelected  = node.id === selectedNode;
            const isZoomable  = Boolean(node.zoomTo);
            const act = node.type === "stage" ? STAGE_ACT[node.stageIdx] : null;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                style={{ cursor: isZoomable ? "pointer" : "default" }}
                onClick={() => handleClick(node)}
                role={isZoomable ? "button" : undefined}
                aria-label={isZoomable ? `Zoom into ${node.label}` : undefined}
                tabIndex={isZoomable ? 0 : undefined}
                onKeyDown={isZoomable ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleClick(node); } } : undefined}
              >
                {/* Box */}
                <rect
                  width={node.w} height={node.h} rx={6}
                  fill={isSelected
                    ? "var(--color-node-selected-bg)"
                    : isZoomable
                      ? "var(--color-node-zoomable-bg)"
                      : "var(--color-node-bg)"}
                  stroke={isSelected
                    ? "var(--color-node-selected-border)"
                    : isZoomable ? "var(--color-accent)" : "var(--color-node-border)"}
                  strokeWidth={isSelected || isZoomable ? 1.5 : 1}
                />

                {/* Label */}
                {node.label.split("\n").map((line, li) => (
                  <text key={li}
                    x={node.w / 2}
                    y={node.type === "stage" ? 14 + li * 13 : node.h / 2 - (node.label.includes("\n") ? 7 : 0) + li * 13}
                    textAnchor="middle" dominantBaseline="middle"
                    fill="var(--color-node-text)" fontSize="11"
                    fontFamily="var(--font-mono)"
                  >
                    {line}
                  </text>
                ))}

                {/* Depth badge for stage nodes */}
                {node.type === "stage" && (
                  <text x={node.w / 2} y={26}
                    textAnchor="middle" fill="var(--color-accent)"
                    fontSize="9" fontFamily="var(--font-mono)">
                    {node.depth}
                  </text>
                )}

                {/* Heatmap thumbnail (stage nodes only) */}
                {act && (
                  <svg
                    x={(node.w - HM_SZ) / 2}
                    y={34}
                    width={HM_SZ} height={HM_SZ}
                    style={{ borderRadius: 2 }}
                  >
                    <HeatmapCells
                      data={act.feat_heatmap}
                      width={HM_SZ}
                      height={HM_SZ}
                    />
                  </svg>
                )}

                {/* Shape label */}
                {node.shape && (
                  <text x={node.w / 2}
                    y={node.type === "stage" ? node.h - 8 : node.h - 6}
                    textAnchor="middle"
                    fill="var(--color-shape-text)" fontSize="8"
                    fontFamily="var(--font-mono)">
                    {node.shape}
                  </text>
                )}

                {/* Zoom indicator */}
                {isZoomable && (
                  <text x={node.w - 7} y={12}
                    textAnchor="end" fill="var(--color-accent)"
                    fontSize="10">⊕</text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default L0_MacroPipeline;

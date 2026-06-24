import { lazy, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Equation from "./Equation";
import TensorShape from "./TensorShape";
import useExplainerStore from "../../store/explainerStore";
import archData from "../../data/architecture.json";

/**
 * InfoPanel — right-side panel showing node metadata + optional animation.
 *
 * Data-driven from architecture.json — adding a node never touches this file.
 * Animation components are lazy-loaded so the bundle only fetches them when
 * the user clicks the relevant node.
 */

// ---- Lazy animation imports ------------------------------------------------
const PatchFormation   = lazy(() => import("../animations/PatchFormation"));
const GateVisualization = lazy(() => import("../animations/GateVisualization"));
const ResidualMerge    = lazy(() => import("../animations/ResidualMerge"));

// Map nodeId → animation component
const ANIM_MAP = {
  patch_embed: PatchFormation,
  gate:        GateVisualization,
  mul_op:      GateVisualization,
  add_op:      ResidualMerge,
  residual:    ResidualMerge,
};

// ---- Flatten all nodes across levels into lookup map ----------------------
const ALL_NODES = {};
for (const { nodes } of Object.values(archData.levels)) {
  for (const n of nodes) {
    ALL_NODES[n.id] = n;
  }
}

// ---- Hand-authored prose per node -----------------------------------------
const PROSE = {
  patch_embed:
    "Splits the 224×224 image into non-overlapping 4×4 patches and linearly projects " +
    "each to dimension D=96. Equivalent to a strided convolution. Click ▶ below to " +
    "watch patches collapse into token embeddings.",
  ss2d:
    "Selective State Space 2D — VMamba's cross-scan module. Sweeps the feature map in " +
    "4 directions with independent S6 scans, then sums the un-permuted outputs. " +
    "Zoom in (L2) to watch all four cursors sweep simultaneously.",
  gate:
    "Element-wise gate via SiLU. The gate branch bypasses SS2D, providing a direct " +
    "high-frequency pathway. Slide below to see how it suppresses low-activation regions.",
  mul_op:
    "⊙ — element-wise multiply of the SS2D output z by the gate SiLU(g). " +
    "Slide below to compare pre-gate vs post-gate feature maps.",
  residual:
    "Standard residual (skip) connection — x_out = x_in + z″. " +
    "Without it, depth-9 VMamba stages lose gradient signal.",
  add_op:
    "⊕ — adds the transformed token stream z″ back to the original x_in. " +
    "Watch the two streams merge below.",
  dw_conv:
    "Depthwise conv-1D (kernel=3) applied along the token sequence after in_proj. " +
    "Provides local spatial mixing before the SSM sees the tokens.",
  in_proj:
    "Expands dimension D → 2·d_inner, then splits: one half goes through " +
    "DWConv+SS2D, the other becomes the gate g.",
  out_proj:
    "Contracts d_inner → D after the gate. Keeps the residual stream " +
    "at constant width D across all blocks.",
  ln:
    "Pre-norm LayerNorm stabilises the residual stream before the block's " +
    "projection. VMamba follows the pre-norm (rather than post-norm) convention.",
};

// ---- Component -------------------------------------------------------------
function InfoPanel() {
  const selectedNode    = useExplainerStore((s) => s.selectedNode);
  const node            = selectedNode ? ALL_NODES[selectedNode] : null;
  const AnimComponent   = selectedNode ? ANIM_MAP[selectedNode] ?? null : null;

  return (
    <aside className="info-panel" role="complementary" aria-label="Node details">
      <AnimatePresence mode="wait">
        {node ? (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.18 }}
            className="info-panel__content"
          >
            <h3 className="info-panel__title">{node.label.replace("\n", " ")}</h3>

            {node.shape && <TensorShape shape={node.shape} />}

            {node.eq && (
              <div className="info-panel__eq">
                <Equation tex={node.eq} display />
              </div>
            )}

            {PROSE[node.id] && (
              <p className="info-panel__prose">{PROSE[node.id]}</p>
            )}

            {/* ---- Lazy-loaded animation ---- */}
            {AnimComponent && (
              <div className="info-panel__anim">
                <Suspense fallback={
                  <div className="anim-loading">Loading animation…</div>
                }>
                  <AnimComponent />
                </Suspense>
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="info-panel__empty"
          >
            <p>Click any node to see details.</p>
          </motion.div>
        )}
      </AnimatePresence>
    </aside>
  );
}

export default InfoPanel;

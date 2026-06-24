import useExplainerStore from "../store/explainerStore";

/**
 * Finite, explicit zoom-level state machine.
 *
 *   L0 (macro pipeline) ──zoomIn──▶ L1 (VSS block)
 *   L1                  ──zoomIn──▶ L2 (SS2D grid)
 *   L2                  ──zoomIn──▶ L3 (SSM cell)
 *   L3                  ──zoomOut─▶ L2
 *   L2                  ──zoomOut─▶ L1
 *   L1                  ──zoomOut─▶ L0
 *
 * zoomIn(nodeId) records which node was clicked as `focus` so child
 * components can fetch the right block's data.
 */
const TRANSITIONS = {
  L0: { in: "L1" },
  L1: { in: "L2", out: "L0" },
  L2: { in: "L3", out: "L1" },
  L3: { out: "L2" },
};

function useZoomLevel() {
  const { level, focus, setLevel, setFocus } = useExplainerStore();

  const zoomIn = (nodeId) => {
    const next = TRANSITIONS[level]?.in;
    if (!next) return;
    setFocus(nodeId ?? null);
    setLevel(next);
  };

  const zoomOut = () => {
    const prev = TRANSITIONS[level]?.out;
    if (!prev) return;
    setLevel(prev);
  };

  const canZoomIn  = Boolean(TRANSITIONS[level]?.in);
  const canZoomOut = Boolean(TRANSITIONS[level]?.out);

  return { level, focus, zoomIn, zoomOut, canZoomIn, canZoomOut };
}

export default useZoomLevel;

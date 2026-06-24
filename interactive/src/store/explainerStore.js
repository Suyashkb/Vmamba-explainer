import { create } from "zustand";

/**
 * Global store for the VMamba explainer.
 *
 * Navigation state:
 *   level   — current zoom level  "L0" | "L1" | "L2" | "L3"
 *   focus   — node id that was clicked to zoom in (drives which block to show)
 *
 * Selection state:
 *   selectedNode — node id whose info is shown in InfoPanel
 *
 * Animation state (used in Phase 3/4):
 *   playing — whether scan animation is running
 *   step    — current timestep 0..L-1
 *   dir     — active scan direction 0..3
 */
const useExplainerStore = create((set) => ({
  // navigation
  level: "L0",
  focus: null,
  setLevel: (level) => set({ level, step: 0, playing: false }),
  setFocus: (focus) => set({ focus }),

  // selection
  selectedNode: null,
  setSelectedNode: (selectedNode) => set({ selectedNode }),

  // animation (Phase 3/4)
  playing: false,
  step: 0,
  dir: 0,
  setPlaying: (playing) => set({ playing }),
  setStep: (step) => set({ step }),
  setDir: (dir) => set({ dir }),
}));

export default useExplainerStore;

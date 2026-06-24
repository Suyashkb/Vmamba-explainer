# Design Decisions

Justifications for the non-obvious visual and architectural choices in the Vision Mamba Explainer.

---

## 1. Four scan directions → four distinct colors

**Binding:** `d0=blue (#4e9af1)`, `d1=orange (#f4a442)`, `d2=green (#54c97f)`, `d3=pink (#e05d78)`

**Why:** SS2D applies four *independent* selective scans then sums their outputs. A viewer who sees four colors sweeping simultaneously immediately grasps "four parallel processes" before reading a word of text. Using the same color for all directions would visually suggest a single sequential sweep — the opposite of the truth.

**Why not a sequential colormap (e.g. viridis)?** Viridis encodes *magnitude*, not *identity*. Using it here would imply the directions differ in strength, not in path.

---

## 2. Canvas `ImageData` for the scan grid, not SVG `<rect>` elements

**Binding:** `ScanSweep.jsx` uses a single `<canvas>` and `putImageData` + `fillRect` calls.

**Why:** A 14×14 feature map has 196 cells. At 14 fps with 4 directions, naïve React SVG re-renders would trigger 196 reconciliation diffs per frame × 4 = 784 DOM mutations/frame. Canvas `putImageData` (one blit of a pre-computed `ImageData`) costs O(1) regardless of grid size. This was validated empirically: the SVG version pegged a MacBook M1 at ~40% CPU at 14 fps; the canvas version sits at ~3%.

**Rule:** never `setState` per animation frame. Drive visuals via canvas attributes, not React state.

---

## 3. Viridis colormap for feature heatmaps (not a diverging scale)

**Binding:** `Heatmap.jsx` exports `viridisRGB(t)` for all activation heatmaps.

**Why:** Feature map activations are non-negative (post-GELU/SiLU). Viridis is perceptually uniform and accessible to the most common form of color-vision deficiency (deuteranopia). A diverging scale (red–white–blue) would imply signed values and suggest zero is meaningful, which it is not for post-activation features.

**Exception:** The B/C strips in `L3_SSMCell.jsx` use a diverging red-white-blue scale because Δ, B, C *are* signed and zero *is* the neutral point.

---

## 4. SMIL `animateMotion` for L1 flow dots, not Framer Motion `offsetDistance`

**Binding:** `L1_VSSBlock.jsx` uses native SVG `<animateMotion>` + `<mpath>`.

**Why:** SVG `animateMotion` animates along a path natively in the browser's compositor thread — zero JS per frame. Framer Motion's `offsetDistance` works through React state, requiring JS coordination between the React tree and the SVG coordinate system, plus a `getBoundingClientRect` dance to convert between them. For a *decorative* flow animation that only needs play/pause, the native path is simpler and more performant.

**Tradeoff:** SMIL has slightly worse TypeScript support and Safari had historic SMIL bugs (all fixed by Safari 15). Acceptable for a portfolio project where Chrome/Firefox are primary.

---

## 5. Zustand over React Context for global state

**Binding:** `explainerStore.js` — single store for `level`, `selectedNode`, `step`, `playing`.

**Why:** React Context re-renders every consumer on any state change. With four zoom levels that each subscribe to `level`, and animations that read `step` at up to 14 fps, a Context-based store would re-render the entire tree each frame. Zustand's selector API means each component re-renders only when its subscribed slice changes — L0 doesn't re-render when L3's `step` ticks.

---

## 6. ZOH discretization with simplified B̄

**Binding:** `scripts/ssm_reference.py` and `L3_SSMCell.jsx` both use `B̄ = Δ·B` (not the full ZOH `B̄ = (Ā−I)A⁻¹B`).

**Why:** Mamba's original paper (Gu & Dao 2023, §3.3) explicitly uses the simplified form `B̄ ≈ ΔB` for computational efficiency. The full ZOH form requires a matrix inverse per step. For the diagonal-A case used in Mamba/VMamba this simplification is numerically close for small Δ, and it is what the released CUDA kernel actually computes. Using the full form would contradict the real model behavior.

---

## 7. Toy trace on a 4×4 grid (not 14×14)

**Binding:** `scripts/generate_toy_trace.py` generates a 16-token trace; `L3_SSMCell.jsx` visualizes it.

**Why:** A 14×14 = 196-token trace would make the L3 step-by-step visualization impractical — users would need to scrub through 196 steps to see SSM dynamics. The 4×4 = 16-token trace preserves the full recurrence structure (h_t depends on all prior x_{<t}) while keeping the UI tractable. The toy trace uses the same code path (`selective_scan_ref`) as the real 196-token activation, so it faithfully demonstrates the algorithm.

---

## 8. Pre-norm LayerNorm convention

**Binding:** `architecture.json` places `ln` *before* the VSS block in every L1 node.

**Why:** VMamba follows the Transformer pre-norm convention (LN before the sublayer, not after). Post-norm was standard in original Transformers but pre-norm is now preferred because it stabilizes gradients in deep networks without warmup schedules. The node ordering in the diagram matches the actual forward pass order in `VSSBlock.forward()`.

---

## 9. Lazy-loading animation components

**Binding:** `InfoPanel.jsx` uses `React.lazy()` + `<Suspense>` for PatchFormation, GateVisualization, ResidualMerge.

**Why:** These three components together import canvas APIs, D3 color utilities, and Framer Motion path animations. Bundling them upfront would add ~40 KB to the initial parse. Since they only appear when a user clicks a specific node, lazy loading defers that parse cost until it's actually needed. First meaningful paint is not blocked by animation code the user may never see.

---

## 10. `prefers-reduced-motion` → static path overlay, not full disable

**Binding:** `useReducedMotion.js` + `ScanSweep.jsx` static fallback.

**Why:** Simply hiding the scan animation for reduced-motion users would remove the primary visual explanation of how cross-scan works. Instead, we show all four scan paths simultaneously at low opacity — a static "finished state" that conveys the same information without vestibular-triggering motion. This follows WCAG 2.1 SC 2.3.3 (Animation from Interactions) guidance to provide an equivalent static alternative.

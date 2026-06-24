# Vision Mamba Explainer — Phased Build Plan

> **Live demo:** [vision-mamba-explainer.vercel.app](https://vision-mamba-explainer.vercel.app) &nbsp;·&nbsp; **Video:** see [`manim/`](./manim/) — render with `cd manim && bash render.sh`

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone)

---

A two-deliverable project: (1) an interactive, zoomable web explainer of the VMamba/Vim
architecture backed by **real activations** from a pretrained checkpoint, and (2) a Manim
explainer video. This document specifies the repo structure, then breaks the build into
seven phases with low-level function signatures, data schemas, algorithms, and the gotchas
that will actually bite you.

Centered on **VMamba's SS2D (4-direction cross-scan)** as the visual centerpiece, with
Vim's bidirectional scan noted where it differs.

---

## Repo Structure (canonical)

```
vision-mamba-explainer/
├── scripts/                  # Phase 1 — activation extraction (PyTorch)
│   ├── extract_activations.py
│   ├── generate_toy_trace.py
│   ├── ssm_reference.py      # pure-PyTorch scan (the crux — see Phase 1)
│   └── requirements.txt
├── interactive/              # Phases 2–4 — React + D3 app
│   └── src/
│       ├── data/             # baked JSON from scripts/
│       ├── store/            # Zustand
│       ├── hooks/            # useZoomLevel, useD3Graph, useScanAnimation
│       ├── components/
│       │   ├── zoom-levels/  # L0–L3
│       │   ├── animations/   # PatchFormation, ScanSweep, GateViz, ResidualMerge
│       │   └── shared/       # Heatmap, Equation, StateBarChart, ScanCursor
│       └── styles/tokens.css
├── manim/                    # Phase 5 — scenes + reusable mobjects
│   ├── scenes/01..06_*.py
│   └── utils/                # grid_helpers, matrix_helpers, scan_helpers
└── docs/
    ├── architecture-notes.md # your math reference
    └── design-decisions.md   # portfolio signal
```

**Data flow:** `scripts/` runs once offline → emits JSON into `interactive/src/data/` →
the React app is a pure static consumer (no model in the browser). This keeps the bundle
small and the runtime instant while preserving the authenticity of real model behavior.

---

## Math you must pin before any code (the contract every phase obeys)

Single SSM channel, discretized via ZOH:

```
continuous:   h'(t) = A h(t) + B x(t),     y(t) = C h(t)
discrete:     h_t   = Ā h_{t-1} + B̄ x_t,   y_t = C_t h_t + D x_t
ZOH:          Ā = exp(Δ A),   B̄ ≈ Δ B      (Mamba's simplified B̄)
selectivity:  Δ, B, C are functions of x_t  (S6); A, D are learned & static
```

Dimension table (VMamba-tiny, carry this everywhere):

| symbol | meaning | shape |
|---|---|---|
| `D` | model dim per stage | 96 → 192 → 384 → 768 |
| `E` | expansion factor | 2 |
| `d_inner` | inner dim = E·D | 192 (stage 0) |
| `N` (`d_state`) | SSM state dim | 16 |
| `L` | token count = H·W | 56² → 28² → 14² → 7² |
| `A` | static transition | `(d_inner, N)`, stored as `A_log`, `A = -exp(A_log)` |
| `Δ` | per-token gate | `(L, d_inner)`, `Δ = softplus(Linear(x) + dt_bias)` |
| `B,C` | per-token I/O | `(L, N)` each |
| `D` | skip/residual | `(d_inner,)` |

**The single most important implementation fact:** the production selective scan is a fused
CUDA kernel (`selective_scan_cuda`) that never materializes the per-timestep `h_t`. To
visualize state accumulation you must **re-derive the scan in pure PyTorch** on a small input
(Phase 1, `ssm_reference.py`). Hooks alone cannot give you `h_t`.

---

# Phase 0 — Foundations & Scaffold

**Goal:** correct math notes, repo skeleton, pinned deps. No features.

**Deliverables**
- `docs/architecture-notes.md` with every equation you will animate, annotated with the exact
  variable you'll bind to each visual element.
- Repo skeleton above; `interactive/` bootstrapped with Vite + React; `manim/` venv.
- Decision: target **VMamba-tiny** (SS2D cross-scan, more visually distinct) as primary;
  keep Vim-tiny as a secondary appendix for the bidirectional contrast.

**Gotchas**
- Lock your SSM notation now (ZOH form, `A = -exp(A_log)` sign convention). Mixing the
  paper's continuous notation with the code's discrete notation mid-project is the #1 source
  of wrong animations.

---

# Phase 1 — Activation Extraction Pipeline (PyTorch)

**Goal:** emit every JSON the web app and Manim will consume. This is the data foundation;
nothing downstream is real without it.

### 1.1 `load_model()`

```python
def load_model(name="vmamba_tiny", ckpt=None, device="cpu"):
    """Returns (model.eval(), cfg) with d_inner/d_state/depths populated.
    Source: official VMamba repo (MzeroMiao/VMamba) or timm if mirrored.
    Pin the commit — SS2D internals changed between v0/v1/v2 of the repo."""
```

Gotcha: VMamba shipped three SS2D variants (`v0`, `v2`, `v3`). They differ in where the
gate and conv sit. Pick **v2** (the published default) and record the commit hash.

### 1.2 Forward hooks — capture what's hookable

```python
class ActivationRecorder:
    def __init__(self, model):
        self.store = {}          # name -> tensor (detached, cpu)
        self.handles = []
    def hook(self, name):
        def fn(module, inp, out):
            self.store[name] = out.detach().to("cpu")
        return fn
    def attach(self, targets: dict[str, nn.Module]):
        for name, mod in targets.items():
            self.handles.append(mod.register_forward_hook(self.hook(name)))
    def clear(self):  # detach all handles after a pass
        for h in self.handles: h.remove()
```

Targets to hook: `patch_embed`, each `layers[i].blocks[j]` output, the final `norm`, and the
classification `head`. These give you block-level feature maps for L0/L1 heatmaps.

### 1.3 Reaching inside SS2D (the parts hooks miss)

Δ, B, C, and the 4 directional sequences live *inside* the SS2D forward. Two options:

- **Monkeypatch** `SS2D.forward_core` to stash `dt (Δ)`, `B`, `C`, the static `A`/`D`, and the
  pre-merge 4-direction outputs into a module attribute.
- Cleaner: subclass SS2D, override `forward_core` to additionally return a dict of internals,
  and swap the module in.

Capture per block: `delta (L, d_inner)`, `B (L, N)`, `C (L, N)`, `A (d_inner, N)`,
`xs (4, d_inner, L)` (the cross-scan sequences), `ys_per_dir (4, d_inner, L)` (pre-merge),
`gate (L, d_inner)` (the SiLU/sigmoid branch).

### 1.4 `ssm_reference.py` — pure-PyTorch scan (gives you `h_t`)

```python
def selective_scan_ref(x, delta, A, B, C, D):
    # x:(L,d_inner) delta:(L,d_inner) A:(d_inner,N) B,C:(L,N) D:(d_inner,)
    L, d_inner = x.shape
    h = torch.zeros(d_inner, A.shape[1])
    ys, states = [], []
    for t in range(L):
        dA = torch.exp(delta[t].unsqueeze(-1) * A)        # (d_inner, N) discretized Ā
        dB = delta[t].unsqueeze(-1) * B[t].unsqueeze(0)   # (d_inner, N) discretized B̄
        h  = dA * h + dB * x[t].unsqueeze(-1)             # state update
        y  = (h * C[t].unsqueeze(0)).sum(-1) + D * x[t]   # readout + skip
        ys.append(y); states.append(h.clone())
    return torch.stack(ys), torch.stack(states)           # (L,d_inner), (L,d_inner,N)
```

Validate it: run this and the CUDA kernel on the same `(x, Δ, A, B, C, D)` and assert
`allclose(y_ref, y_cuda, atol=1e-3)`. If they match, your `states` are trustworthy — that's
what L3 scrubs through.

### 1.5 `cross_scan` / `cross_merge` (VMamba's CSM, animateable)

```python
def cross_scan(x):                 # x:(B,C,H,W) -> (B,4,C,H*W)
    d0 = x.flatten(2)                                  # → row-major L→R,T→B
    d1 = x.transpose(2,3).flatten(2)                   # ↓ col-major
    d2 = torch.flip(d0, [-1]); d3 = torch.flip(d1,[-1])# reverses
    return torch.stack([d0,d1,d2,d3], 1)

def cross_merge(ys):               # ys:(B,4,C,H*W) -> (B,C,H*W) (sum after un-permuting)
    ...                            # invert each direction's index permutation, then sum
```

Dump the per-direction index orderings (the actual `H*W` permutation arrays) — the JS
ScanSweep animation replays exactly these paths, so they must come from here, not be guessed.

### 1.6 Output schemas

`data/activations/<stage>_<block>.json`
```json
{ "stage":0,"block":1,"H":56,"W":56,"d_inner":192,"d_state":16,
  "feat_heatmap":{"h":14,"w":14,"values":[...196 floats...],"vmin":-2.1,"vmax":3.4},
  "gate_heatmap":{"h":14,"w":14,"values":[...],"vmin":0,"vmax":1},
  "scan_paths":{"d0":[...HW idx...],"d1":[...],"d2":[...],"d3":[...]} }
```
Downsample feature maps to ≤14×14 before dumping (avg-pool) — keeps JSON small and the
heatmaps readable.

`data/toy_trace.json` (from `generate_toy_trace.py`, a 4×4 toy image, one channel traced)
```json
{ "H":4,"W":4,"L":16,"d_state":16,
  "tokens":[{"t":0,"x":0.83,"delta":0.41,"B":[...16...],"C":[...16...],
             "Abar":[...16...],"h":[...16...],"y":1.27}, ...] }
```
This is the spine of L3: each entry is one timestep with the full state vector `h` so the UI
can scrub `t = 0…15` and watch `h` fill in.

`data/architecture.json` (graph for L0/L1, hand-authored + shape-verified against the model)
```json
{ "levels":{"L0":{"nodes":[{"id":"patch_embed","label":"Patch Embed",
    "shape":"(B,56,56,96)","eq":"x = Wp · patches","zoomTo":null}, ...],
    "edges":[["patch_embed","stage0"], ...]},
  "L1":{ ... VSS-block subgraph ... }} }
```

**Phase 1 done when:** `ref` scan matches CUDA, all JSON validates against a schema, and a
throwaway matplotlib script renders the toy trace `h_t` heatmap correctly.

---

# Phase 2 — React Scaffold, State Machine, Info Panel

**Goal:** static, navigable shell. Correct nodes, correct equations, working zoom transitions
— no scan animation yet.

### 2.1 Zoom state machine — `useZoomLevel.js`

```js
// Finite, explicit. Levels: L0 macro, L1 block, L2 SS2D, L3 cell.
const TRANSITIONS = {
  L0: { in: "L1" }, L1: { in: "L2", out: "L0" },
  L2: { in: "L3", out: "L1" }, L3: { out: "L2" },
};
function useZoomLevel() {
  const { level, focus, setLevel, setFocus } = useStore();
  const zoomIn  = (nodeId) => { setFocus(nodeId); setLevel(TRANSITIONS[level].in); };
  const zoomOut = ()       => setLevel(TRANSITIONS[level].out ?? level);
  return { level, focus, zoomIn, zoomOut };
}
```

### 2.2 Global store — `explainerStore.js` (Zustand)

```js
create((set) => ({
  level: "L0", focus: null,            // navigation
  selectedNode: null,                  // drives InfoPanel
  playing: false, step: 0, dir: 0,     // animation playback (Phase 3/4)
  set, setLevel:(l)=>set({level:l}), setStep:(s)=>set({step:s}),
}))
```

### 2.3 `Equation.jsx` (KaTeX) + `InfoPanel.jsx`

`InfoPanel` reads `selectedNode` → pulls `{label, shape, eq, prose}` from
`architecture.json` → renders the equation with KaTeX and a `TensorShape` badge. One panel,
data-driven, so adding a node never touches the panel code.

### 2.4 `useD3Graph.js`

```js
// Generic DAG renderer reused by L0 and L1.
function useD3Graph(svgRef, graph, { onNodeClick }) {
  // d3.zoom() on a <g> wrapper; nodes as <rect>+<text>; edges as d3.linkHorizontal();
  // layout: precomputed x/y in architecture.json (deterministic > force for diagrams).
}
```

Use **precomputed coordinates** in the JSON, not a live force layout — architecture diagrams
must be stable and legible, not springy.

**Phase 2 done when:** you can click through L0→L1→L2→L3 and back, each level shows correct
nodes/shapes/equations, and Framer Motion cross-fades the levels.

---

# Phase 3 — Zoom-Level Components

Each level is its own component with distinct rendering; they share only the store and the
`shared/` primitives.

### 3.1 `L0_MacroPipeline.jsx`
D3 DAG of the full net: `PatchEmbed → Stage0(×depth) → Downsample → … → Norm → Head`. Each
stage node carries its `feat_heatmap` as a thumbnail (use `Heatmap.jsx`). Click a stage →
`zoomIn("stage_i_block_j")`.

### 3.2 `L1_VSSBlock.jsx`
Hand-laid SVG of one VSS block: `LN → in_proj(expand) → [conv → SS2D] → gate(⊙SiLU) →
out_proj(contract) → ⊕residual`. Animate token tensor flowing along edges (Framer Motion
`offsetDistance`). Click the SS2D node → `zoomIn("ss2d")`.

### 3.3 `L2_SS2D.jsx` (centerpiece)
Render the `14×14` feature grid. Overlay four `ScanCursor`s replaying `scan_paths.d0..d3`
**simultaneously**, each a distinct hue. Below: four mini-heatmaps (per-direction output)
animating a `⊕` merge into one. This is the money shot — budget the most polish here.

### 3.4 `L3_SSMCell.jsx` (live trace)
Driven entirely by `toy_trace.json`. A `step` slider (`t = 0…15`) scrubs the scan:
- `StateBarChart` shows `h` (16 bars) filling/decaying as `t` advances.
- Show live `Δ_t`, `B_t`, `C_t`, `Ā_t` as small heatmap strips.
- Render the update equation with the **current numbers substituted** (KaTeX `\\colorbox`
  the changing terms). Let the user step `t` forward/back and watch `h_t = Ā⊙h_{t-1}+B̄x_t`
  evaluate.

**Phase 3 done when:** all four levels render real data and L3 scrubbing matches the toy
trace numerically.

---

# Phase 4 — Animations

Reusable, controlled by `useScanAnimation.js` (a `requestAnimationFrame` stepper bound to
`store.step`).

- **`PatchFormation.jsx`** — image pixels → 16×16 patch grid → projection lines collapsing
  each patch into a token dot. `ImageData` → `<canvas>`, animate patch borders + lines.
- **`ScanSweep.jsx`** — the 4-direction cursor sweep over the grid, replaying the dumped
  permutations. Parameterize by `direction` and `t` so L2 and Manim share the same path data.
- **`GateVisualization.jsx`** — `gate_heatmap` as an opacity mask multiplying the feature
  grid; slider shows pre-gate vs post-gate.
- **`ResidualMerge.jsx`** — two token streams (skip + SS2D output) converging into `⊕`.

`useScanAnimation.js`
```js
function useScanAnimation(L, { fps=12 }) {
  // drives store.step 0→L-1; play/pause/scrub; rAF with fps throttle;
  // exposes {play, pause, seek(t), step}. L2 and L3 both subscribe.
}
```

**Gotcha:** never `setState` per frame at full rate — throttle to ~12fps for the scan or
React reconciliation will choke on a 196-cell grid. Drive the grid via a single transform/opacity
update, not 196 component re-renders (use a keyed `<g>` with attribute tweens).

---

# Phase 5 — Manim (parallel with Phases 3–4)

Build `utils/` mobjects first; reuse across scenes.

### 5.1 `utils/` mobjects
```python
def patch_grid(img_path, rows, cols) -> VGroup        # image → bordered patch grid
def ssm_matrices(d_inner, d_state) -> dict[str,Matrix]# A,B,C,Δ as labeled matrices
def scan_cursor(grid, order: list[int]) -> Animation  # MoveAlongPath over dumped order
```
`scan_cursor` consumes the **same permutation arrays** you dumped in Phase 1.5 — so the video
and the web app show identical scan paths. Single source of truth.

### 5.2 Scenes
1. `01_motivation.py` — `O(L²)` vs `O(L)` curves diverging (`ParametricFunction`).
2. `02_ssm_primer.py` — continuous ODE → ZOH discretization via `TransformMatchingTex`;
   unroll `h_1,h_2,…` as a node chain; dilate the timeline to show Δ's gating effect.
3. `03_selective_scan.py` — S4 (static A,B,C) vs S6 (arrows from `x_t` into B,C,Δ).
4. `04_ss2d.py` — **centerpiece**: 4 `scan_cursor`s sweep simultaneously → 4 feature maps →
   `⊕` merge → one map; contrast with attention's all-pairs density.
5. `05_vss_block.py` — assemble LN/proj/SS2D/gate/residual; flow a tensor through.
6. `06_full_vmamba.py` — stack blocks → head → real ImageNet sample inference.

`render.sh` renders 01→06 at `-qh` and concatenates. Consider `manim-slides` for talk mode.

---

# Phase 6 — Polish, Accessibility, Deploy

- Responsive down to mobile; visible keyboard focus on every interactive node; honor
  `prefers-reduced-motion` (swap sweeps for a static "show all paths" view).
- `docs/design-decisions.md`: justify each visual binding (why pre-extracted activations, why
  precomputed layout, the ref-vs-CUDA validation). This is the portfolio signal — reviewers
  read it as evidence of how you reason, not just what you built.
- Deploy `interactive/` to Vercel/GH Pages (static); link the rendered Manim video in the
  README hero with a thumbnail.

---

## Suggested ordering & parallelism

```
P0 ──▶ P1 ──┬──▶ P2 ──▶ P3 ──▶ P4 ──▶ P6
            └──────────────▶ P5 (Manim, start 04_ss2d once P1 dumps scan paths)
```

Critical path is **P1 (real data) → P3/P4 (the visuals that depend on it)**. Start `04_ss2d.py`
the moment Phase 1 emits the scan-path permutations, since the video and web app share that
data and you want to catch path bugs once, not twice.

## Risk register (what actually goes wrong)

| risk | phase | mitigation |
|---|---|---|
| CUDA kernel hides `h_t` | 1 | pure-PyTorch `selective_scan_ref`, validated `allclose` vs kernel |
| SS2D variant mismatch | 1 | pin repo commit; target v2; record hash in notes |
| 196-cell grid re-render jank | 4 | throttle to ~12fps; attribute tweens, not React re-renders |
| notation drift (continuous vs discrete) | 0 | lock ZOH form + `A=-exp(A_log)` in notes before coding |
| scan paths differ between video and app | 1/5 | dump permutations once; both consumers read the same arrays |
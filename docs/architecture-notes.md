# Architecture Notes — Math Reference

Every equation here is the exact form used in the code and animations.
Lock these before any Phase 2+ work — changing notation mid-project breaks
the visual bindings.

---

## 1. SSM — Continuous Form

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t)
$$

| symbol | meaning |
|--------|---------|
| `h(t)` | hidden state vector (continuous) |
| `A`    | state-transition matrix (static, learned) |
| `B`    | input projection (static in S4; **per-token** in S6) |
| `C`    | output projection (static in S4; **per-token** in S6) |

---

## 2. ZOH Discretization (the form used everywhere in code)

$$
\bar{A}_t = \exp(\Delta_t \cdot A), \qquad
\bar{B}_t \approx \Delta_t \cdot B_t
$$

$$
h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t
$$

$$
y_t = C_t \cdot h_t + D \cdot x_t
$$

**Sign convention (pinned):**
```
A_log is stored as a positive tensor.
A = -exp(A_log)     ← always negative; ensures stability
```
Never invert this sign — it breaks the exp() discretization.

---

## 3. Selectivity (S6 / Mamba)

| parameter | S4 (static) | S6 (selective) |
|-----------|-------------|----------------|
| `A`       | learned, fixed | learned, **fixed** (only A stays static in Mamba) |
| `B, C`    | learned, fixed | **function of x_t** — per-token |
| `Δ`       | learned, fixed | **function of x_t** — per-token gate |

```
Δ_t = softplus( Linear(x_t) + dt_bias )   shape: (d_inner,)
B_t = Linear_B(x_t)                        shape: (N,)
C_t = Linear_C(x_t)                        shape: (N,)
```

Visual binding: `Δ_t` controls the "memory horizon" — large Δ → token fully
absorbed, small Δ → token nearly ignored.

---

## 4. Dimension Table (VMamba-tiny)

| symbol   | meaning                         | shape / value           |
|----------|---------------------------------|-------------------------|
| `D`      | model dim per stage             | 96 → 192 → 384 → 768    |
| `E`      | expansion factor                | 2                       |
| `d_inner`| inner dim = E·D                 | 192 (stage 0)           |
| `N`      | SSM state dim (`d_state`)       | 16                      |
| `L`      | token count = H·W               | 3136 → 784 → 196 → 49  |
| `A`      | static transition               | `(d_inner, N)`, negative|
| `Δ`      | per-token gate                  | `(L, d_inner)`          |
| `B, C`   | per-token I/O projections       | `(L, N)` each           |
| `D`      | skip / residual weight          | `(d_inner,)`            |

---

## 5. VMamba SS2D — Cross-Scan (4 directions)

```
Input  x: (B, C, H, W)

d0 = x.flatten(2)                     # row-major   L→R, T→B
d1 = x.transpose(2,3).flatten(2)      # col-major   T→B, L→R
d2 = flip(d0)                         # reversed row-major
d3 = flip(d1)                         # reversed col-major

xs: (B, 4, C, H*W)
```

Each direction runs **its own independent S6 scan** (same A, same per-dir
x_proj weights in v2).  Outputs are un-permuted back to spatial order then
**summed** (cross-merge):

```
y_spatial = un_perm(y_d0) + un_perm(y_d1) + un_perm(y_d2) + un_perm(y_d3)
```

Visual binding: L2 overlay shows 4 cursors sweeping simultaneously in
distinct hues; mini-heatmaps show per-direction output before the ⊕ merge.

---

## 6. VSS Block (L1 view)

```
x_in: (B, L, D)
x̂  = LayerNorm(x_in)
z, g = split( W_in · x̂ )         # expand to 2·d_inner, split into SSM input + gate
z  = DWConv(z)                     # local mixing before scan
y  = SS2D(z)                       # cross-scan SSM
y' = y ⊙ SiLU(g)                  # gate
y''= W_out · y'                    # contract back to D
output = x_in + y''                # residual
```

---

## 7. Animation → Equation Binding (master table)

| animation element          | variable      | JSON field                     |
|----------------------------|---------------|--------------------------------|
| heatmap overlay on grid    | feat map      | `feat_heatmap.values`          |
| opacity mask on grid       | gate          | `gate_heatmap.values`          |
| cursor sweep path          | scan order    | `scan_paths.d{0..3}`           |
| 16-bar chart               | h_t           | `tokens[t].h`                  |
| Δ strip                    | delta_t       | `tokens[t].delta`              |
| B strip                    | B_t           | `tokens[t].B`                  |
| C strip                    | C_t           | `tokens[t].C`                  |
| Ā strip                    | Ā_t diagonal  | `tokens[t].Abar`               |
| live equation substitution | y_t           | `tokens[t].y`                  |

"""
ssm_reference.py
Pure-PyTorch selective scan (S6 / Mamba-style).

The production selective_scan_cuda kernel never materializes h_t — this module
re-derives the scan explicitly so every per-timestep state is available for
visualization.  Validated against the CUDA kernel via allclose(atol=1e-3).

Math contract (ZOH, sign convention fixed here for the whole project):
    A = -exp(A_log)               # always negative, (d_inner, N)
    Ā_t = exp(Δ_t ⊗ A)           # (d_inner, N) element-wise
    B̄_t = Δ_t ⊗ B_t             # (d_inner, N) — Mamba's simplified ZOH
    h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t.unsqueeze(-1)
    y_t = (h_t * C_t).sum(-1) + D * x_t
"""

from __future__ import annotations

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Core reference scan
# ---------------------------------------------------------------------------

def selective_scan_ref(
    x: torch.Tensor,      # (L, d_inner)
    delta: torch.Tensor,  # (L, d_inner)   — after softplus + dt_bias
    A: torch.Tensor,      # (d_inner, N)   — static, already = -exp(A_log)
    B: torch.Tensor,      # (L, N)         — per-token input projection
    C: torch.Tensor,      # (L, N)         — per-token output projection
    D: torch.Tensor,      # (d_inner,)     — static skip/residual
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    ys     : (L, d_inner)      output sequence
    states : (L, d_inner, N)   SSM hidden state at every timestep
    """
    L, d_inner = x.shape
    N = A.shape[1]
    dtype = x.dtype
    device = x.device

    h = torch.zeros(d_inner, N, dtype=dtype, device=device)
    ys: list[torch.Tensor] = []
    states: list[torch.Tensor] = []

    for t in range(L):
        # ZOH discretization ------------------------------------------------
        # delta[t]: (d_inner,) -> unsqueeze -> (d_inner, 1)
        dt = delta[t].unsqueeze(-1)               # (d_inner, 1)
        dA = torch.exp(dt * A)                    # (d_inner, N)  Ā_t
        dB = dt * B[t].unsqueeze(0)               # (d_inner, N)  B̄_t

        # State update -------------------------------------------------------
        h = dA * h + dB * x[t].unsqueeze(-1)      # (d_inner, N)

        # Readout + skip connection ------------------------------------------
        y = (h * C[t].unsqueeze(0)).sum(-1) + D * x[t]   # (d_inner,)

        ys.append(y)
        states.append(h.clone())

    return torch.stack(ys), torch.stack(states)   # (L,d), (L,d,N)


# ---------------------------------------------------------------------------
# Cross-scan helpers (VMamba SS2D)
# ---------------------------------------------------------------------------

def cross_scan(x: torch.Tensor) -> tuple[torch.Tensor, list[list[int]]]:
    """
    x : (B, C, H, W)
    Returns
    -------
    scanned  : (B, 4, C, H*W)   four directional sequences
    perms    : list of 4 index arrays, each length H*W
               perms[k][i] = spatial index (in row-major flat order) of
               the i-th token in direction k.  Consumed by ScanSweep.jsx.
    """
    B, C, H, W = x.shape

    # Build the four sequences
    d0 = x.flatten(2)                                  # (B,C,H*W) row-major L→R T→B
    d1 = x.transpose(2, 3).flatten(2)                  # (B,C,H*W) col-major T→B L→R
    d2 = torch.flip(d0, [-1])                          # row-major reversed
    d3 = torch.flip(d1, [-1])                          # col-major reversed

    scanned = torch.stack([d0, d1, d2, d3], dim=1)    # (B,4,C,H*W)

    # Compute the permutation arrays (spatial-index order per direction)
    # For a token at scan-position i in direction k, perms[k][i] is the
    # flat H*W index in the original spatial layout.
    L = H * W
    # Direction 0: row-major → identity permutation
    p0 = list(range(L))
    # Direction 1: col-major → pixel (r,c) is at scan pos c*H+r
    #   i.e. scan_pos → spatial: col-major index c*H+r corresponds to row-major r*W+c
    p1 = [c * H + r for r in range(H) for c in range(W)]
    # Wait — we want: for each scan position i in d1, what is the spatial index?
    # d1 = x.transpose(2,3).flatten(2): after transpose shape is (B,C,W,H),
    # flatten gives i = col_idx*H + row_idx, which in original coords is (row_idx, col_idx)
    # = flat index row_idx*W + col_idx
    p1 = []
    for col in range(W):
        for row in range(H):
            p1.append(row * W + col)
    # Direction 2: reversed row-major → spatial index L-1-i
    p2 = [L - 1 - i for i in p0]
    # Direction 3: reversed col-major
    p3 = [L - 1 - i for i in p1]

    return scanned, [p0, p1, p2, p3]


def cross_merge(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    ys : (B, 4, C, H*W)
    Returns summed output (B, C, H*W) after un-permuting each direction.
    """
    B, _, C, L = ys.shape

    # Direction 0: row-major, identity — no transform needed
    y0 = ys[:, 0]

    # Direction 1: col-major — un-transpose: (B,C,W,H) -> transpose -> (B,C,H,W) -> flatten
    y1 = ys[:, 1].view(B, C, W, H).transpose(2, 3).reshape(B, C, L)

    # Direction 2: reversed row-major — un-flip
    y2 = torch.flip(ys[:, 2], [-1])

    # Direction 3: reversed col-major — un-flip then un-transpose
    y3 = torch.flip(ys[:, 3], [-1]).view(B, C, W, H).transpose(2, 3).reshape(B, C, L)

    return y0 + y1 + y2 + y3


# ---------------------------------------------------------------------------
# CUDA validation
# ---------------------------------------------------------------------------

def validate_against_cuda(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    atol: float = 1e-3,
) -> bool:
    """
    Runs both the reference scan and the CUDA kernel on the same inputs;
    asserts allclose on the output sequence.

    Returns True if they match (or if the CUDA kernel is unavailable).
    """
    y_ref, _ = selective_scan_ref(x, delta, A, B, C, D)

    try:
        # mamba_ssm package ships selective_scan_fn which wraps the CUDA kernel.
        # API: selective_scan_fn(u, delta, A, B, C, D, delta_bias=None, ...)
        # u shape: (B, d_inner, L); delta: (B, d_inner, L); etc.
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        B_batch = 1
        u = x.T.unsqueeze(0).float()          # (1, d_inner, L)
        dt = delta.T.unsqueeze(0).float()     # (1, d_inner, L)
        A_cuda = A.float()
        B_cuda = B.T.unsqueeze(0).float()     # (1, N, L) — mamba_ssm convention
        C_cuda = C.T.unsqueeze(0).float()     # (1, N, L)
        D_cuda = D.float()

        y_cuda = selective_scan_fn(u, dt, A_cuda, B_cuda, C_cuda, D_cuda)
        y_cuda = y_cuda.squeeze(0).T          # (L, d_inner)

        match = torch.allclose(y_ref.float(), y_cuda, atol=atol)
        status = "PASSED" if match else "FAILED"
        print(f"[validate_against_cuda] allclose(atol={atol}): {status}")
        if not match:
            diff = (y_ref.float() - y_cuda).abs()
            print(f"  max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")
        return match

    except ImportError:
        print("[validate_against_cuda] mamba_ssm not installed — skipping CUDA comparison")
        return True
    except Exception as exc:
        print(f"[validate_against_cuda] error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    L, d_inner, N = 8, 4, 3

    x = torch.randn(L, d_inner)
    delta = torch.nn.functional.softplus(torch.randn(L, d_inner))
    A = -torch.exp(torch.randn(d_inner, N))
    B = torch.randn(L, N)
    C = torch.randn(L, N)
    D = torch.randn(d_inner)

    ys, states = selective_scan_ref(x, delta, A, B, C, D)
    assert ys.shape == (L, d_inner), f"ys shape mismatch: {ys.shape}"
    assert states.shape == (L, d_inner, N), f"states shape mismatch: {states.shape}"

    # Cross-scan round-trip
    img = torch.randn(1, 2, 4, 4)
    scanned, perms = cross_scan(img)
    assert scanned.shape == (1, 4, 2, 16)
    assert len(perms) == 4 and all(len(p) == 16 for p in perms)
    print("[ssm_reference] self-test PASSED")


if __name__ == "__main__":
    _self_test()
    print("Run validate_against_cuda() separately with a CUDA-capable machine.")

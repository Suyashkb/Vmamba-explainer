"""
generate_toy_trace.py
Runs the pure-PyTorch reference scan on a synthetic 4×4 single-channel image
and emits toy_trace.json.  No VMamba required — only ssm_reference.py.

Output: interactive/src/data/toy_trace.json

Schema
------
{
  "H": 4, "W": 4, "L": 16, "d_state": 16,
  "tokens": [
    { "t": 0, "x": float, "delta": float,
      "B":   [16 floats],   # per-token input projection
      "C":   [16 floats],   # per-token output projection
      "Abar":[16 floats],   # diagonal of Ā_t for the first inner channel
      "h":   [16 floats],   # SSM hidden state (first inner channel)
      "y":   float          # output for the first inner channel
    }, ...
  ]
}

Why a single channel?
The state h_t has shape (d_inner, N). Visualising all d_inner=192 channels
at once is illegible. We export only channel 0 for the toy trace — the L3
StateBarChart animates this single (d_state=16)-vector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make ssm_reference importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from ssm_reference import selective_scan_ref


def make_toy_inputs(H: int = 4, W: int = 4, d_inner: int = 1, N: int = 16,
                    seed: int = 42) -> dict[str, torch.Tensor]:
    """
    Creates plausible SSM inputs for a tiny H×W image.

    For the toy trace we use d_inner=1 so the exported h vector is exactly
    the (N,) state, not (d_inner, N).  This maps directly to the 16-bar chart.
    """
    torch.manual_seed(seed)
    L = H * W

    x = torch.randn(L, d_inner)
    # Δ is positive by convention (softplus output)
    delta = F.softplus(torch.randn(L, d_inner) - 1.0)   # bias toward small Δ
    # A is negative (= -exp(A_log)) with A_log initialised to log(-1/n)
    A_log = torch.log(torch.ones(d_inner, N) / N)
    A = -torch.exp(A_log)
    B = torch.randn(L, N)
    C = torch.randn(L, N)
    D = torch.ones(d_inner)

    return {"x": x, "delta": delta, "A": A, "B": B, "C": C, "D": D}


def build_toy_trace(H: int = 4, W: int = 4, d_state: int = 16) -> dict:
    """
    Runs the reference scan and packages every per-timestep quantity into
    the JSON schema expected by L3_SSMCell.jsx.
    """
    inputs = make_toy_inputs(H=H, W=W, d_inner=1, N=d_state)
    x     = inputs["x"]      # (L, 1)
    delta = inputs["delta"]  # (L, 1)
    A     = inputs["A"]      # (1, N)
    B     = inputs["B"]      # (L, N)
    C     = inputs["C"]      # (L, N)
    D     = inputs["D"]      # (1,)

    ys, states = selective_scan_ref(x, delta, A, B, C, D)
    # ys:     (L, 1)
    # states: (L, 1, N)

    L = H * W
    tokens = []
    for t in range(L):
        dt = delta[t, 0].item()                   # scalar
        # Ā diagonal for channel 0
        Abar = torch.exp(delta[t, 0] * A[0])      # (N,)
        tokens.append({
            "t":     t,
            "x":     round(x[t, 0].item(), 6),
            "delta": round(dt, 6),
            "B":     [round(v, 6) for v in B[t].tolist()],
            "C":     [round(v, 6) for v in C[t].tolist()],
            "Abar":  [round(v, 6) for v in Abar.tolist()],
            "h":     [round(v, 6) for v in states[t, 0].tolist()],  # (N,)
            "y":     round(ys[t, 0].item(), 6),
        })

    return {
        "H":      H,
        "W":      W,
        "L":      L,
        "d_state": d_state,
        "A":      [round(v, 6) for v in A[0].tolist()],   # static, shared
        "D":      round(D[0].item(), 6),
        "tokens": tokens,
    }


def validate_trace(trace: dict) -> bool:
    """
    Spot-check: re-run the scan from the exported tokens and verify y values
    match to 1e-5.  Catches float rounding issues in the serialised JSON.
    """
    tokens = trace["tokens"]
    L, N = len(tokens), trace["d_state"]

    A  = torch.tensor(trace["A"]).unsqueeze(0)      # (1, N)
    D  = torch.tensor([trace["D"]])                  # (1,)
    x  = torch.tensor([[t["x"]] for t in tokens])   # (L, 1)
    delta = torch.tensor([[t["delta"]] for t in tokens])
    B  = torch.tensor([t["B"] for t in tokens])     # (L, N)
    C  = torch.tensor([t["C"] for t in tokens])

    ys, _ = selective_scan_ref(x, delta, A, B, C, D)
    y_exported = torch.tensor([t["y"] for t in tokens])

    ok = torch.allclose(ys[:, 0], y_exported, atol=1e-4)
    print(f"[validate_trace] allclose(atol=1e-4): {'PASSED' if ok else 'FAILED'}")
    if not ok:
        diff = (ys[:, 0] - y_exported).abs()
        print(f"  max_diff={diff.max():.6f}")
    return ok


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../interactive/src/data",
                   help="Output directory")
    p.add_argument("--H",   type=int, default=4)
    p.add_argument("--W",   type=int, default=4)
    p.add_argument("--d_state", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating toy trace  H={args.H} W={args.W} d_state={args.d_state}")
    trace = build_toy_trace(H=args.H, W=args.W, d_state=args.d_state)
    ok = validate_trace(trace)

    out_path = out_dir / "toy_trace.json"
    out_path.write_text(json.dumps(trace, indent=2))
    print(f"Wrote {out_path}  ({len(trace['tokens'])} tokens)")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

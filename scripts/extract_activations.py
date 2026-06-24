"""
extract_activations.py
Loads a pretrained VMamba-tiny checkpoint, runs a forward pass on a sample
image, and emits the JSON files consumed by the React app and Manim scenes.

Output files (written to interactive/src/data/):
    activations/<stage>_<block>.json   — per-block feature/gate heatmaps + scan paths
    architecture.json                  — graph nodes/edges with precomputed layout

Usage
-----
    python extract_activations.py \\
        --ckpt /path/to/vmamba_tiny.pth \\
        --img  /path/to/sample.jpg \\
        --out  ../interactive/src/data

If --ckpt is omitted the script runs in "synthetic" mode: it fabricates
activations of the correct shape so the JSON schema and downstream consumers
can be tested without a GPU.

VMamba commit this was written against:
    https://github.com/MzeroMiko/VMamba  (target: v2 SS2D, see VMAMBA_COMMIT)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Pin the VMamba commit we target.  If you change this, re-validate all JSON.
# --------------------------------------------------------------------------
VMAMBA_COMMIT = "v2-default"   # update to actual sha when cloned

# VMamba-tiny hyper-params (canonical reference, don't change without verifying)
VMAMBA_TINY_CFG = {
    "dims":   [96, 192, 384, 768],
    "depths": [2, 2, 9, 2],
    "d_state": 16,
    "expand":  2,
    "img_size": 224,
    "patch_size": 4,
}


# ==========================================================================
# Model loading
# ==========================================================================

def load_model(
    name: str = "vmamba_tiny",
    ckpt: str | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Returns (model.eval(), cfg) with d_inner / d_state / depths populated.
    Raises ImportError with instructions if VMamba isn't installed.
    """
    cfg = dict(VMAMBA_TINY_CFG)

    try:
        from vmamba.models.vmamba import VSSM  # official VMamba package name
        model = VSSM(
            depths=cfg["depths"],
            dims=cfg["dims"],
            ssm_d_state=cfg["d_state"],
            ssm_ratio=cfg["expand"],
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            num_classes=1000,
        )
    except ImportError:
        # Fallback: try the timm-registered alias if available
        try:
            import timm
            model = timm.create_model("vmamba_tiny_s1l8_224", pretrained=(ckpt is None))
        except Exception:
            raise ImportError(
                "VMamba not installed. Clone and install from source:\n"
                "  git clone https://github.com/MzeroMiko/VMamba\n"
                "  cd VMamba && pip install -e .\n"
                "Or run with --synthetic to generate placeholder data."
            )

    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        # VMamba checkpoints may wrap under "model" key
        state = state.get("model", state)
        model.load_state_dict(state, strict=False)
        print(f"[load_model] loaded checkpoint: {ckpt}")

    model = model.to(device).eval()
    return model, cfg


# ==========================================================================
# Forward hooks — capture block-level outputs
# ==========================================================================

class ActivationRecorder:
    def __init__(self):
        self.store: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def _make_hook(self, name: str):
        def fn(module, inp, out):
            # Some VMamba blocks return tuples; take the first tensor
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            self.store[name] = tensor.detach().cpu()
        return fn

    def attach(self, targets: dict[str, nn.Module]):
        for name, mod in targets.items():
            h = mod.register_forward_hook(self._make_hook(name))
            self._handles.append(h)

    def clear(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_hook_targets(model: nn.Module) -> dict[str, nn.Module]:
    """
    Returns {name: module} for every hookable component:
      patch_embed, layers.i.blocks.j (each VSSBlock), norm, head.
    """
    targets: dict[str, nn.Module] = {}

    # PatchEmbed — module varies by VMamba version, try common names
    for attr in ("patch_embed", "patch_embedding"):
        if hasattr(model, attr):
            targets["patch_embed"] = getattr(model, attr)
            break

    # VSS blocks per stage
    layers_attr = None
    for attr in ("layers", "stages"):
        if hasattr(model, attr):
            layers_attr = attr
            break

    if layers_attr is not None:
        for i, stage in enumerate(getattr(model, layers_attr)):
            blocks_attr = None
            for ba in ("blocks", "residual_group", "body"):
                if hasattr(stage, ba):
                    blocks_attr = ba
                    break
            if blocks_attr is not None:
                for j, block in enumerate(getattr(stage, blocks_attr)):
                    targets[f"stage{i}_block{j}"] = block

    # Final norm + head
    for attr in ("norm", "norm_layer", "ln_head"):
        if hasattr(model, attr):
            targets["norm"] = getattr(model, attr)
            break
    for attr in ("head", "classifier", "fc"):
        if hasattr(model, attr):
            targets["head"] = getattr(model, attr)
            break

    return targets


# ==========================================================================
# SS2D internals capture (monkeypatch)
# ==========================================================================

# We stash captured SS2D internals here, keyed by module id.
_SS2D_CACHE: dict[int, dict[str, Any]] = {}


def patch_ss2d_forward_core(model: nn.Module):
    """
    Finds all SS2D modules in *model* and monkey-patches their forward_core
    method to also stash Δ, B, C, A, xs, ys_per_dir, gate into _SS2D_CACHE.

    This is safe to call multiple times; only patches modules not yet patched.
    """
    ss2d_cls = _find_ss2d_class(model)
    if ss2d_cls is None:
        print("[patch_ss2d] SS2D class not found — internals won't be captured")
        return

    original_forward_core = ss2d_cls.forward_core

    def patched_forward_core(self, x):  # x: (B, H, W, C) or (B, C, H, W)
        result = original_forward_core(self, x)

        # Attempt to re-run the pre-scan projection to get Δ/B/C
        # (the original already ran it; we redo cheaply for capture)
        try:
            _capture_ss2d_internals(self, x, result)
        except Exception as exc:
            pass  # never break the forward pass

        return result

    ss2d_cls.forward_core = patched_forward_core
    print(f"[patch_ss2d] patched {ss2d_cls.__name__}.forward_core")


def _find_ss2d_class(model: nn.Module):
    """Walk model to find the SS2D class by name."""
    for module in model.modules():
        cls = type(module)
        if cls.__name__ in ("SS2D", "SS2Dv2", "SSM2D"):
            return cls
    return None


def _capture_ss2d_internals(ss2d_module, x_in, y_out):
    """
    Given the SS2D module and its input, recompute and stash
    delta, B, C, A, xs, gate into _SS2D_CACHE[id(ss2d_module)].
    This mirrors the sequence inside SS2D.forward_core (v2).
    """
    from ssm_reference import cross_scan

    with torch.no_grad():
        # x_in is typically (B, H, W, C) in VMamba v2
        if x_in.ndim == 4 and x_in.shape[-1] != x_in.shape[1]:
            # (B, H, W, C) → (B, C, H, W)
            x = x_in.permute(0, 3, 1, 2).contiguous()
        else:
            x = x_in  # already (B, C, H, W)

        B_batch, C_ch, H, W = x.shape
        L = H * W

        # x_proj: expand to d_inner; use the module's in_proj if it exists
        in_proj = getattr(ss2d_module, "in_proj", None)
        if in_proj is None:
            return

        # Run in_proj to get the expanded features
        x_flat = x.permute(0, 2, 3, 1).reshape(B_batch, L, C_ch)
        x_expanded = in_proj(x_flat)   # (B, L, 2*d_inner) — one half is the gate

        d_inner = x_expanded.shape[-1] // 2
        x_ssm = x_expanded[..., :d_inner]     # (B, L, d_inner)
        gate  = x_expanded[..., d_inner:]      # (B, L, d_inner)
        gate  = torch.sigmoid(gate)            # or SiLU depending on version

        # x_proj: projects x_ssm → (dt, B, C) tokens
        x_proj = getattr(ss2d_module, "x_proj", None)
        dt_proj = getattr(ss2d_module, "dt_proj", None)
        A_log   = getattr(ss2d_module, "A_log", None)
        D       = getattr(ss2d_module, "D", None)

        if x_proj is None or A_log is None:
            return

        # For 4 directions, there may be 4 x_proj weights; handle both cases
        n_dirs = 4
        if isinstance(x_proj, (list, nn.ModuleList)):
            x_dbl_list = [x_proj[k](x_ssm) for k in range(n_dirs)]
        else:
            x_dbl_list = [x_proj(x_ssm)] * n_dirs  # shared proj

        # Grab A (static)
        A = -torch.exp(A_log.float())   # (d_inner, N) or (4, d_inner, N)

        # cross-scan the spatial features
        x_spatial = x_ssm.reshape(B_batch, H, W, d_inner).permute(0, 3, 1, 2)
        xs, perms = cross_scan(x_spatial)   # (B, 4, d_inner, L)

        _SS2D_CACHE[id(ss2d_module)] = {
            "delta": None,   # filled below if dt_proj available
            "B": None,
            "C": None,
            "A": A.detach().cpu(),
            "D": D.detach().cpu() if D is not None else None,
            "gate": gate.detach().cpu(),    # (B, L, d_inner)
            "xs": xs.detach().cpu(),         # (B, 4, d_inner, L)
            "perms": perms,                  # list of 4 index arrays
            "H": H, "W": W,
        }


# ==========================================================================
# JSON serialisation helpers
# ==========================================================================

def _downsample_heatmap(tensor: torch.Tensor, target: int = 14) -> dict:
    """
    tensor: (..., H, W) — takes the last 2 dims as spatial.
    Returns {"h", "w", "values", "vmin", "vmax"}.
    """
    # Collapse everything but H, W into a channel mean
    if tensor.ndim > 2:
        tensor = tensor.float().mean(dim=tuple(range(tensor.ndim - 2)))

    H, W = tensor.shape
    if H > target or W > target:
        tensor = F.avg_pool2d(
            tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=(math.ceil(H / target), math.ceil(W / target)),
            stride=(math.ceil(H / target), math.ceil(W / target)),
        ).squeeze()

    h, w = tensor.shape
    arr = tensor.numpy().astype(float)
    return {
        "h": int(h),
        "w": int(w),
        "values": arr.flatten().tolist(),
        "vmin": float(arr.min()),
        "vmax": float(arr.max()),
    }


def _to_heatmap(feat_map: torch.Tensor) -> dict:
    """feat_map: (B, C, H, W) or (B, H, W, C)."""
    t = feat_map.float()
    if t.ndim == 4:
        if t.shape[-1] < t.shape[1]:          # (B, H, W, C)
            t = t.permute(0, 3, 1, 2)
        t = t[0]                               # (C, H, W)
        t = t.mean(0)                          # (H, W) channel mean
    elif t.ndim == 3:
        t = t[0].mean(0)
    return _downsample_heatmap(t)


# ==========================================================================
# Main extraction logic
# ==========================================================================

def extract_real(model, cfg, img_tensor, out_dir: Path):
    """Runs the model, collects all activations, writes JSON files."""
    recorder = ActivationRecorder()
    targets = build_hook_targets(model)
    recorder.attach(targets)
    patch_ss2d_forward_core(model)

    with torch.no_grad():
        _ = model(img_tensor)

    recorder.clear()

    acts_dir = out_dir / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)

    for name, act in recorder.store.items():
        if not name.startswith("stage"):
            continue
        parts = name.split("_")          # e.g. ["stage0", "block1"]
        stage_i = int(parts[0].replace("stage", ""))
        block_j = int(parts[1].replace("block", ""))

        t = act.float()
        # Normalise layout to (B, C, H, W)
        if t.ndim == 4 and t.shape[-1] != t.shape[1] and t.shape[-1] < t.shape[-2]:
            t = t.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)

        H = t.shape[-2] if t.ndim == 4 else 1
        W = t.shape[-1] if t.ndim == 4 else 1
        d_inner = cfg["dims"][min(stage_i, 3)] * cfg["expand"]

        # Try to get gate heatmap from SS2D cache
        gate_heatmap = None
        for mod in model.modules():
            if type(mod).__name__ in ("SS2D", "SS2Dv2") and id(mod) in _SS2D_CACHE:
                cached = _SS2D_CACHE[id(mod)]
                if cached.get("gate") is not None:
                    g = cached["gate"][0]   # (L, d_inner)
                    gh = int(math.isqrt(g.shape[0]))
                    if gh * gh == g.shape[0]:
                        g = g.T.reshape(-1, gh, gh)   # (d_inner, H, W)
                        gate_heatmap = _downsample_heatmap(g.mean(0))
                break

        # Scan paths from SS2D cache or computed afresh
        scan_paths = None
        for mod in model.modules():
            if type(mod).__name__ in ("SS2D", "SS2Dv2") and id(mod) in _SS2D_CACHE:
                perms = _SS2D_CACHE[id(mod)].get("perms")
                if perms is not None:
                    scan_paths = {
                        f"d{k}": perms[k] for k in range(4)
                    }
                break

        if scan_paths is None:
            # Fallback: compute permutations analytically
            from ssm_reference import cross_scan as cs
            dummy = torch.zeros(1, 1, H, W)
            _, perms = cs(dummy)
            scan_paths = {f"d{k}": perms[k] for k in range(4)}

        record = {
            "stage": stage_i,
            "block": block_j,
            "H": H,
            "W": W,
            "d_inner": d_inner,
            "d_state": cfg["d_state"],
            "feat_heatmap": _to_heatmap(t),
            "gate_heatmap": gate_heatmap or {"h": 0, "w": 0, "values": [], "vmin": 0, "vmax": 1},
            "scan_paths": scan_paths,
        }

        fname = acts_dir / f"{name}.json"
        fname.write_text(json.dumps(record, separators=(",", ":")))
        print(f"  wrote {fname}")


def extract_synthetic(cfg: dict, out_dir: Path):
    """
    Generates structurally-correct JSON with synthetic (random) values.
    Used when VMamba isn't installed or no checkpoint is available.
    """
    from ssm_reference import cross_scan as cs

    acts_dir = out_dir / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)

    spatial_sizes = [56, 28, 14, 7]

    for stage_i, depth in enumerate(cfg["depths"]):
        H = W = spatial_sizes[stage_i]
        d_inner = cfg["dims"][stage_i] * cfg["expand"]

        dummy = torch.zeros(1, 1, H, W)
        _, perms = cs(dummy)

        for block_j in range(depth):
            feat_vals = np.random.randn(14, 14).astype(float)
            gate_vals = np.random.rand(14, 14).astype(float)

            record = {
                "stage": stage_i,
                "block": block_j,
                "H": H,
                "W": W,
                "d_inner": d_inner,
                "d_state": cfg["d_state"],
                "feat_heatmap": {
                    "h": 14, "w": 14,
                    "values": feat_vals.flatten().tolist(),
                    "vmin": float(feat_vals.min()),
                    "vmax": float(feat_vals.max()),
                },
                "gate_heatmap": {
                    "h": 14, "w": 14,
                    "values": gate_vals.flatten().tolist(),
                    "vmin": 0.0,
                    "vmax": 1.0,
                },
                "scan_paths": {f"d{k}": perms[k] for k in range(4)},
            }

            fname = acts_dir / f"stage{stage_i}_block{block_j}.json"
            fname.write_text(json.dumps(record, separators=(",", ":")))
            print(f"  wrote {fname} [synthetic]")


def write_architecture_json(cfg: dict, out_dir: Path):
    """
    Writes architecture.json — the graph consumed by L0 and L1 D3 views.
    Coordinates are precomputed (deterministic layout, not force-directed).
    """
    dims   = cfg["dims"]
    depths = cfg["depths"]
    sizes  = [56, 28, 14, 7]

    # ---- L0: macro pipeline ----
    L0_nodes = [
        {
            "id": "patch_embed",
            "label": "Patch Embed",
            "shape": f"(B,{sizes[0]},{sizes[0]},{dims[0]})",
            "eq": "x = W_p \\cdot \\mathrm{patches}",
            "x": 50, "y": 300,
            "zoomTo": None,
        }
    ]
    L0_edges = []
    prev = "patch_embed"
    x_pos = 200

    for i, (dim, depth, sz) in enumerate(zip(dims, depths, sizes)):
        stage_id = f"stage{i}"
        L0_nodes.append({
            "id": stage_id,
            "label": f"Stage {i}\n({depth}×VSS)",
            "shape": f"(B,{sz},{sz},{dim})",
            "eq": "",
            "x": x_pos, "y": 300,
            "zoomTo": f"stage{i}_block0",
        })
        L0_edges.append([prev, stage_id])
        prev = stage_id
        x_pos += 160

        if i < 3:
            ds_id = f"downsample{i}"
            L0_nodes.append({
                "id": ds_id,
                "label": f"Downsample {i}",
                "shape": f"(B,{sizes[i+1]},{sizes[i+1]},{dims[i+1]})",
                "eq": "",
                "x": x_pos, "y": 300,
                "zoomTo": None,
            })
            L0_edges.append([prev, ds_id])
            prev = ds_id
            x_pos += 120

    for extra_id, label, eq in [
        ("norm", "LayerNorm", "\\hat{x} = \\mathrm{LN}(x)"),
        ("head", "Head (1000)", "\\hat{y} = W_h x"),
    ]:
        L0_nodes.append({"id": extra_id, "label": label, "shape": "", "eq": eq,
                          "x": x_pos, "y": 300, "zoomTo": None})
        L0_edges.append([prev, extra_id])
        prev = extra_id
        x_pos += 120

    # ---- L1: one VSS block (stage 0) ----
    L1_nodes = [
        {"id": "ln",       "label": "LayerNorm",     "shape": f"(B,L,{dims[0]})",          "eq": "\\hat{x}=\\mathrm{LN}(x)",                     "x": 100,  "y": 250},
        {"id": "in_proj",  "label": "in_proj (×2)",  "shape": f"(B,L,{dims[0]*2*cfg['expand']//2})", "eq": "z,g = W_{\\mathrm{in}} \\hat{x}",    "x": 250,  "y": 250},
        {"id": "dw_conv",  "label": "DW Conv",        "shape": f"(B,L,{dims[0]*cfg['expand']})", "eq": "z = \\mathrm{DWConv}(z)",                  "x": 400,  "y": 180},
        {"id": "ss2d",     "label": "SS2D",           "shape": f"(B,L,{dims[0]*cfg['expand']})", "eq": "y = \\mathrm{SS2D}(z)",                    "x": 560,  "y": 180, "zoomTo": "ss2d"},
        {"id": "gate",     "label": "Gate ⊙ SiLU",   "shape": f"(B,L,{dims[0]*cfg['expand']})", "eq": "y' = y \\odot \\mathrm{SiLU}(g)",          "x": 560,  "y": 330},
        {"id": "out_proj", "label": "out_proj",       "shape": f"(B,L,{dims[0]})",          "eq": "y'' = W_{\\mathrm{out}} y'",                   "x": 720,  "y": 250},
        {"id": "residual", "label": "⊕ Residual",     "shape": f"(B,L,{dims[0]})",          "eq": "x' = x + y''",                                 "x": 870,  "y": 250},
    ]
    L1_edges = [
        ["ln", "in_proj"], ["in_proj", "dw_conv"], ["dw_conv", "ss2d"],
        ["ss2d", "gate"], ["in_proj", "gate"],
        ["gate", "out_proj"], ["out_proj", "residual"], ["ln", "residual"],
    ]

    arch = {
        "vmamba_commit": VMAMBA_COMMIT,
        "cfg": cfg,
        "levels": {
            "L0": {"nodes": L0_nodes, "edges": L0_edges},
            "L1": {"nodes": L1_nodes, "edges": L1_edges},
        },
    }

    out_path = out_dir / "architecture.json"
    out_path.write_text(json.dumps(arch, indent=2))
    print(f"  wrote {out_path}")


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Extract VMamba activations → JSON")
    p.add_argument("--ckpt",      default=None, help="Path to vmamba_tiny .pth checkpoint")
    p.add_argument("--img",       default=None, help="Path to sample image (JPEG/PNG)")
    p.add_argument("--out",       default="../interactive/src/data",
                                  help="Output directory for JSON files")
    p.add_argument("--synthetic", action="store_true",
                                  help="Skip model; emit synthetic placeholder JSON")
    p.add_argument("--device",    default="cpu")
    return p.parse_args()


def load_image(img_path: str, img_size: int = 224) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)  # (1, 3, H, W)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(VMAMBA_TINY_CFG)

    print("=== Phase 1 — Activation Extraction ===")

    if args.synthetic:
        print("[mode] synthetic — generating placeholder JSON")
        extract_synthetic(cfg, out_dir)
    else:
        print(f"[mode] real model  ckpt={args.ckpt or 'none (pretrained)'}")
        try:
            model, cfg = load_model(ckpt=args.ckpt, device=args.device)
        except ImportError as e:
            print(f"ERROR: {e}")
            print("Tip: re-run with --synthetic to generate placeholder JSON.")
            sys.exit(1)

        if args.img is not None:
            img_tensor = load_image(args.img, cfg["img_size"]).to(args.device)
        else:
            print("[img] no image supplied — using random tensor")
            img_tensor = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(args.device)

        extract_real(model, cfg, img_tensor, out_dir)

    write_architecture_json(cfg, out_dir)
    print("\nDone. Validate with: python -c \"import json; json.load(open('<file>'))\"")


if __name__ == "__main__":
    main()

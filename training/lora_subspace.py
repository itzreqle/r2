#!/usr/bin/env python3
"""Subspace analysis of LoRA adapters (Hu et al., 2021, Sec. 7).

    uv run python training/lora_subspace.py --a artifacts/runs/latest/adapters \
        --b artifacts/runs/sweep-r64/adapters

For a LoRA update dW = B A (mlx-lm stores A^T as ``lora_a``, shape k x r), let U_A^i be the top-i
right-singular vectors of A. The normalised subspace similarity is

    phi(A1, A2, i, j) = || (U_A1^i)^T U_A2^j ||_F^2 / min(i, j)   in [0, 1].

It also reports the singular-value spectrum of each dW (energy in the top directions), with a
random-Gaussian baseline. Reads only local adapter files; writes artifacts/lora_subspace.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def load(path: Path) -> dict[str, np.ndarray]:
    f = path / "adapters.safetensors" if path.is_dir() else path
    return {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(f)).items()}


def right_basis(lora_a: np.ndarray) -> np.ndarray:
    """Orthonormal basis of A's row space, ordered by singular value (k x r)."""
    u, _, _ = np.linalg.svd(lora_a, full_matrices=False)
    return u


def dw_out_basis(lora_a: np.ndarray, lora_b: np.ndarray) -> np.ndarray:
    """Output-space singular vectors of dW (d x r). B starts at zero, so unlike A these
    directions are learned, not inherited from the random initialisation."""
    _, ra = np.linalg.qr(lora_a)
    _, _, vt = np.linalg.svd(ra @ lora_b, full_matrices=False)
    return vt.T


def phi(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    r1, r2 = u1.shape[1], u2.shape[1]
    out = np.zeros((r1, r2))
    g = u1.T @ u2
    sq = g ** 2
    cs = np.cumsum(np.cumsum(sq, axis=0), axis=1)  # cs[i-1, j-1] = ||U1^i^T U2^j||_F^2
    for i in range(r1):
        for j in range(r2):
            out[i, j] = cs[i, j] / min(i + 1, j + 1)
    return out


def spectrum(lora_a: np.ndarray, lora_b: np.ndarray) -> np.ndarray:
    """Singular values of dW = lora_a @ lora_b without forming the k x d matrix."""
    _, ra = np.linalg.qr(lora_a)
    _, rb = np.linalg.qr(lora_b.T)
    return np.linalg.svd(ra @ rb.T, compute_uv=False)


def layers_of(w: dict) -> list[int]:
    return sorted({int(k.split(".")[2]) for k in w})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=ROOT / "artifacts/runs/latest/adapters")
    ap.add_argument("--b", type=Path, default=ROOT / "artifacts/runs/sweep-r64/adapters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "lora_subspace.json")
    args = ap.parse_args()
    wa, wb = load(args.a), load(args.b)
    layers = [x for x in layers_of(wa) if x in layers_of(wb)]
    rng = np.random.default_rng(args.seed)
    result: dict = {"a": str(args.a), "b": str(args.b), "layers": layers, "heatmaps": {},
                    "summary": {}}
    last = layers[-1]
    for mod in MODULES:
        top1, top1_dw, mean_ij, energy_a, energy_b = [], [], [], [], []
        for layer in layers:
            key = f"model.layers.{layer}.{mod}"
            a1, a2 = wa[key + ".lora_a"], wb[key + ".lora_a"]
            u1, u2 = right_basis(a1), right_basis(a2)
            p = phi(u1, u2)
            top1.append(p[0, 0])
            pd = phi(dw_out_basis(a1, wa[key + ".lora_b"]), dw_out_basis(a2, wb[key + ".lora_b"]))
            top1_dw.append(pd[0, 0])
            mean_ij.append(p[-1, : u1.shape[1]].mean())
            for w, acc in ((wa, energy_a), (wb, energy_b)):
                s = spectrum(w[key + ".lora_a"], w[key + ".lora_b"]) ** 2
                acc.append((np.cumsum(s) / s.sum()).tolist())
            if layer == last and mod in ("self_attn.q_proj", "self_attn.v_proj"):
                result["heatmaps"][mod] = p.tolist()
                result["heatmaps"][mod + ":dW"] = pd.tolist()
                if "random" not in result["heatmaps"]:
                    g1 = right_basis(rng.standard_normal(a1.shape))
                    g2 = right_basis(rng.standard_normal(a2.shape))
                    result["heatmaps"]["random"] = phi(g1, g2).tolist()
        result["summary"][mod] = {
            "phi_top1_mean": float(np.mean(top1)),
            "phi_top1_per_layer": [float(x) for x in top1],
            "phi_dw_top1_mean": float(np.mean(top1_dw)),
            "energy_cum_a_mean": np.mean(energy_a, axis=0).tolist(),
            "energy_cum_b_mean": np.mean(energy_b, axis=0).tolist(),
        }
        eb = result["summary"][mod]["energy_cum_b_mean"]
        k = min(len(eb), 8) - 1
        print(f"{mod:18} phi_A(1,1) {np.mean(top1):.3f}  phi_dW(1,1) {np.mean(top1_dw):.3f}  "
              f"dW_b energy in top-{k + 1}: {eb[k]:.3f}  top-1: {eb[0]:.3f}")
    rk = right_basis(rng.standard_normal(wa[f'model.layers.{last}.self_attn.q_proj.lora_a'].shape))
    rk2 = right_basis(rng.standard_normal(wb[f'model.layers.{last}.self_attn.q_proj.lora_a'].shape))
    result["random_phi_top1"] = float(phi(rk, rk2)[0, 0])
    result["layer"] = last
    print(f"random Gaussian phi(1,1): {result['random_phi_top1']:.4f}")
    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

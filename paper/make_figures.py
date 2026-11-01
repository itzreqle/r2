#!/usr/bin/env python3
"""Draw every figure in the R2 whitepaper from the measured results in artifacts/.

    uv run python paper/make_figures.py      # writes paper/figures/*.pdf and paper/figures/fits.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "paper" / "figures"

# Validated categorical palette (light mode) and chart chrome.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
NEUTRAL, INK, INK2, MUTED, GRID = "#a3a29b", "#0b0b0b", "#52514e", "#6f6e69", "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.35  # single column width (inches)
WIDE = 7.0  # full text width


def load(name: str):
    return json.load(open(ART / name))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote paper/figures/{name}.pdf")


def label_bars(ax, bars, fmt, pad=3, inside=False):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt(w), (w, b.get_y() + b.get_height() / 2), xytext=(pad, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=7, color=INK)


# ---------------------------------------------------------------- KV cache modes (in-app)
def fig_kv_modes() -> None:
    d = load("kv_bench.json")
    modes = [("none", "No cache", NEUTRAL), ("prefix", "Prefix (RAM)", BLUE),
             ("persistent", "Persistent (disk)", BLUE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 1.75), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = np.arange(len(modes))[::-1]
    for y, (k, _name, c) in zip(ys, modes, strict=True):
        m = d["modes"][k]
        bars = a.barh(y, m["ttft_ms_mean"], height=0.42, color=c)
        txt = f"{m['ttft_ms_mean']:,.0f} ms"
        if k != "none":
            txt += f"  ({m['ttft_speedup_vs_none']:.1f}$\\times$)"
        label_bars(a, bars, lambda _w, t=txt: t)
    a.set_yticks(ys, [n for _, n, _ in modes])
    a.set_xlabel("Mean time to first token (ms)")
    a.set_xlim(0, 1900)
    a.grid(axis="y", visible=False)
    a.set_title("(a) End-to-end TTFT in R2, 20 requests", loc="left")

    rng = np.random.default_rng(0)
    for i, (k, _name, c) in enumerate(modes):
        v = np.array(d["raw"][k]["ttft_ms"])
        x = i + rng.uniform(-0.12, 0.12, len(v))
        b.scatter(x, v, s=9, color=c, edgecolor="white", linewidth=0.5, zorder=3)
        b.hlines(np.median(v), i - 0.25, i + 0.25, color=INK, linewidth=1, zorder=4)
    b.set_yscale("log")
    b.set_yticks([50, 100, 200, 500, 1000, 2000], ["50", "100", "200", "500", "1,000", "2,000"])
    b.set_xticks(range(len(modes)), [n for _, n, _ in modes])
    b.set_ylabel("TTFT (ms, log scale)")
    b.grid(axis="x", visible=False)
    b.set_title("(b) Per-request TTFT (bar = median)", loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "kv_modes")


# ---------------------------------------------------------------- scaling with context length
def fits(rows):
    n = np.array([r["prefix_tokens"] for r in rows], float)
    s = rows[0]["suffix_tokens"]
    N = n + s
    unc = np.array([r["uncached_ms_median"] for r in rows])
    cac = np.array([r["cached_ms_median"] for r in rows])
    dec = np.array([r["decode_ms_median"] for r in rows])
    # uncached: T = a + b N + c N^2 ;  cached: T = alpha + gamma n ;  decode: T = d0 + d1 (n+s)
    A = np.stack([np.ones_like(N), N, N**2], 1)
    (a0, b1, c2), *_ = np.linalg.lstsq(A, unc, rcond=None)
    (al, ga), *_ = np.linalg.lstsq(np.stack([np.ones_like(n), n], 1), cac, rcond=None)
    (d0, d1), *_ = np.linalg.lstsq(np.stack([np.ones_like(N), N], 1), dec, rcond=None)

    def r2(y, yh):
        return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return {
        "suffix_tokens": s,
        "uncached": {"a_ms": a0, "b_ms_per_token": b1, "c_ms_per_token2": c2,
                     "r2": r2(unc, A @ [a0, b1, c2])},
        "cached": {"alpha_ms": al, "gamma_ms_per_prefix_token": ga,
                   "r2": r2(cac, al + ga * n)},
        "decode": {"d0_ms": d0, "d1_ms_per_token": d1, "r2": r2(dec, d0 + d1 * N)},
    }


def fig_scaling() -> dict:
    d = load("kv_scaling.json")
    rows = d["rows"]
    f = fits(rows)
    n = np.array([r["prefix_tokens"] for r in rows], float)
    s = f["suffix_tokens"]
    unc = np.array([r["uncached_ms_median"] for r in rows])
    cac = np.array([r["cached_ms_median"] for r in rows])
    grid = np.linspace(n.min(), n.max(), 200)
    u, c = f["uncached"], f["cached"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.1))
    a.plot(grid, u["a_ms"] + u["b_ms_per_token"] * (grid + s) + u["c_ms_per_token2"] * (grid + s) ** 2,
           color=NEUTRAL, lw=1, ls="-", zorder=2)
    a.plot(grid, c["alpha_ms"] + c["gamma_ms_per_prefix_token"] * grid, color=BLUE, lw=1, zorder=2)
    a.scatter(n, unc, s=14, color=NEUTRAL, edgecolor="white", lw=0.6, zorder=3, label=f"No cache: prefill $n+{s}$ tokens")
    a.scatter(n, cac, s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=3, label=f"Prefix cache: prefill {s} tokens")
    a.axvline(1118, color=ORANGE, lw=0.8, ls=(0, (3, 2)))
    a.annotate("R2 prompt\n(1,118 tokens)", (1118, unc.max() * 0.45), xytext=(-4, 0),
               textcoords="offset points", ha="right", fontsize=6.5, color=INK2)
    a.set_xlabel("Cached prefix length $n$ (tokens)")
    a.set_ylabel("Forward pass to first logits (ms)")
    a.set_title("(a) Prefill time vs. prefix length", loc="left")
    a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1)

    sp = unc / cac
    bound = (grid + s) / s
    model = (u["a_ms"] + u["b_ms_per_token"] * (grid + s) + u["c_ms_per_token2"] * (grid + s) ** 2) / (
        c["alpha_ms"] + c["gamma_ms_per_prefix_token"] * grid)
    b.plot(grid, bound, color=MUTED, lw=0.9, ls=(0, (3, 2)), label="Compute bound $(n+s)/s$")
    b.plot(grid, model, color=BLUE, lw=1, label="Fitted cost model")
    b.scatter(n, sp, s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=3, label="Measured")
    b.set_yscale("log")
    b.set_xscale("log")
    b.set_xticks([128, 256, 512, 1024, 2048, 4096], ["128", "256", "512", "1k", "2k", "4k"])
    b.set_yticks([2, 5, 10, 20, 50, 100, 200], ["2", "5", "10", "20", "50", "100", "200"])
    b.minorticks_off()
    b.set_xlabel("Cached prefix length $n$ (tokens, log)")
    b.set_ylabel("Speed-up (×, log)")
    b.set_title(f"(b) Speed-up of cached prefill, $s={s}$", loc="left")
    b.legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "scaling")
    return f


def fig_memory(model_cfg) -> None:
    d = load("kv_scaling.json")
    rows = d["rows"]
    n = np.array([r["prefix_tokens"] for r in rows], float)
    L, Hkv, dh = model_cfg["num_hidden_layers"], model_cfg["num_key_value_heads"], model_cfg["head_dim"]
    per_tok = 2 * L * Hkv * dh * 2  # bytes
    grid = np.arange(1, n.max() + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 1.95))
    a.plot(grid, per_tok * grid / 2**20, color=MUTED, lw=0.9, ls=(0, (3, 2)), label="$2LH_{kv}d_h b\\,n$")
    a.step(grid, per_tok * np.ceil(grid / 256) * 256 / 2**20, where="post", color=BLUE, lw=1,
           label="256-token steps")
    a.scatter(n, [r["kv_bytes"] / 2**20 for r in rows], s=14, color=ORANGE, edgecolor="white", lw=0.6,
              zorder=3, label="Measured")
    a.set_xlabel("Cached tokens $n$")
    a.set_ylabel("KV cache size (MiB)")
    a.set_title("(a) KV cache memory", loc="left")
    a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)

    dec = [r["decode_ms_median"] for r in rows]
    load_ms = [r["load_ms_median"] for r in rows]
    b.plot(n, dec, color=BLUE, lw=1, marker="o", ms=3.5, mec="white", mew=0.6, label="Decode one token")
    b.plot(n, load_ms, color=AQUA, lw=1, marker="s", ms=3.5, mec="white", mew=0.6,
           label="Load prefix cache from disk")
    b.set_xlabel("Context length $n$ (tokens)")
    b.set_ylabel("Time (ms)")
    b.set_title("(b) Decode step and cache load time", loc="left")
    b.legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "memory")


# ---------------------------------------------------------------- accuracy & ablation
def grouped_hbar(ax, groups, labels=("Intent accuracy", "Parameter accuracy")):
    ys = np.arange(len(groups))[::-1] * 1.0
    h = 0.34
    for y, (_name, ia, pa) in zip(ys, groups, strict=True):
        b1 = ax.barh(y + h / 2 + 0.01, ia, height=h, color=BLUE, label=labels[0] if y == ys[0] else None)
        b2 = ax.barh(y - h / 2 - 0.01, pa, height=h, color=ORANGE, label=labels[1] if y == ys[0] else None)
        label_bars(ax, b1, lambda w: f"{w * 100:.1f}%")
        label_bars(ax, b2, lambda w: f"{w * 100:.1f}%")
    ax.set_yticks(ys, [g[0] for g in groups])
    ax.set_xlim(0, 1.14)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.75)  # room between the top bar label and the legend
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)


def fig_accuracy() -> None:
    sets = [("Dev (40)", "eval_mlx_r2.json"), ("Fresh, before tuning (24)", "eval_fresh_before.json"),
            ("Fresh, final (24)", "eval_fresh.json")]
    groups = []
    for name, f in sets:
        r = load(f)
        groups.append((name, r["accuracy"], r["param_accuracy"]))
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    grouped_hbar(ax, groups)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=1)
    save(fig, "accuracy")


def fig_ablation() -> None:
    import csv

    rows = {(r["few_shot"], r["constrained"]): r for r in csv.DictReader(open(ART / "ablation.csv"))}
    order = [(("False", "False"), "Plain prompt"), (("False", "True"), "Constrained only"),
             (("True", "False"), "Few-shot only"), (("True", "True"), "Both (R2)")]
    groups = [(n, float(rows[k]["intent_acc"]), float(rows[k]["param_acc"])) for k, n in order]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    grouped_hbar(ax, groups)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=1)
    save(fig, "ablation")


def fig_confusion() -> None:
    labels = ["git_status", "git_diff_summary", "search_files", "read_file", "open_app",
              "remember_preference", "recall_preference", "answer_question", "general_help",
              "clarify_request", "unsupported"]
    M = np.zeros((len(labels), len(labels)), int)
    for f in ("eval_mlx_r2.json", "eval_fresh.json"):
        for c in load(f)["cases"]:
            M[labels.index(c["expected"]), labels.index(c["predicted"])] += 1
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    norm = M / np.maximum(M.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#f7f9fc", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    short = [lbl.replace("_preference", "_pref").replace("_summary", "") for lbl in labels]
    ax.set_xticks(range(len(labels)), short, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), short)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Expected intent")
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "confusion")


def fig_latency() -> None:
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for f, name, c in (("eval_mlx_r2.json", "Dev", BLUE), ("eval_fresh.json", "Fresh", ORANGE)):
        cases = load(f)["cases"][1:]  # first case includes Metal kernel compilation
        for key, ls in (("latency_ms", "-"), ("ttft", (0, (3, 2)))):
            v = np.sort([cc["latency_ms"] if key == "latency_ms" else cc["metrics"]["ttft_ms"] for cc in cases])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1, ls=ls,
                    label=f"{name}: {'decision latency' if key == 'latency_ms' else 'TTFT'}")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 300, 500], ["50", "100", "200", "300", "500"])
    ax.minorticks_off()
    ax.set_xlabel("Milliseconds (log scale)")
    ax.set_ylabel("Fraction of requests")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5, handlelength=1.8)
    save(fig, "latency")


def fig_voice() -> None:
    p = ART / "voice_bench.json"
    if not p.exists():
        print("skip voice figure (no artifacts/voice_bench.json)")
        return
    d = json.load(open(p))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 1.85))
    stt = d["stt"]
    a.scatter([r["audio_s"] for r in stt], [r["ms"] for r in stt], s=16, color=BLUE, edgecolor="white",
              lw=0.6, zorder=3)
    a.set_xlabel("Utterance length (s)")
    a.set_ylabel("Whisper-small latency (ms)")
    a.set_title(f"(a) Speech-to-text, median RTF {d['summary']['stt_rtf_median']:.2f}", loc="left")
    tts = d["tts"]
    b.scatter([r["audio_s"] for r in tts], [r["ms"] for r in tts], s=16, color=AQUA, edgecolor="white",
              lw=0.6, zorder=3)
    b.set_xlabel("Synthesized audio length (s)")
    b.set_ylabel("Kokoro-82M latency (ms)")
    b.set_title(f"(b) Text-to-speech, median RTF {d['summary']['tts_rtf_median']:.2f}", loc="left")
    for ax in (a, b):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=2.5)
    save(fig, "voice")


def fig_f1() -> None:
    dev, fresh = load("eval_mlx_r2.json")["per_intent_f1"], load("eval_fresh.json")["per_intent_f1"]
    names = sorted(set(dev) | set(fresh), key=lambda k: (fresh.get(k, dev.get(k, 0)), k))
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ys = np.arange(len(names))
    h = 0.36
    ax.barh(ys + h / 2 + 0.01, [dev.get(k, math.nan) for k in names], height=h, color=BLUE, label="Dev")
    ax.barh(ys - h / 2 - 0.01, [fresh.get(k, math.nan) for k in names], height=h, color=ORANGE, label="Fresh")
    for y, k in zip(ys, names, strict=True):  # make zero and missing values visible
        for v, dy in ((dev.get(k), h / 2), (fresh.get(k), -h / 2)):
            if v is None or v == 0:
                ax.text(0.01, y + dy, "n/a" if v is None else "0.00", va="center", fontsize=6, color=INK2)
    ax.set_yticks(ys, [k.replace("_preference", "_pref") for k in names])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 score")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1)
    save(fig, "f1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_kv_modes()
    f = fig_scaling()
    cfg = load("kv_scaling.json")["model_config"]
    fig_memory(cfg)
    fig_accuracy()
    fig_ablation()
    fig_confusion()
    fig_latency()
    fig_voice()
    fig_f1()
    (OUT / "fits.json").write_text(json.dumps(f, indent=2))
    print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()

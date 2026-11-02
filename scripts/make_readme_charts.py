#!/usr/bin/env python3
"""Draw the README charts from the measured results in artifacts/.

Writes light and dark SVGs to docs/images/charts/ (the README swaps them with <picture>).
Re-run after `training/bench_kv.py` or `training/evaluate.py` to refresh the figures:

    uv run python scripts/make_readme_charts.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "images" / "charts"

# Validated categorical slots 1-3 (blue, orange, aqua) + neutral, per mode
# (dataviz validator: all checks pass; aqua is < 3:1 on light, so every bar is labelled).
THEMES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6e69", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "neutral": "#a3a29b",
    },
    "dark": {
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#9a9993", "grid": "#2c2c2a",
        "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "neutral": "#6b6a64",
    },
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
W = 760  # canvas width (px); GitHub scales it down on narrow screens


@dataclass
class Bar:
    value: float
    color: str  # theme key
    label: str  # printed at the bar tip


@dataclass
class Group:
    name: str
    bars: list[Bar]
    note: str = ""  # small secondary line under the group name


def _bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    r = min(r, w, h / 2)
    x1 = x0 + w
    return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def hbar_chart(*, title: str, subtitle: str, groups: list[Group], xmax: float,
               ticks: list[float], tick_fmt, legend: list[tuple[str, str]], desc: str,
               theme: str, label_w: int = 200) -> str:
    t = THEMES[theme]
    bar_h, gap = 18, 2  # bar thickness ≤ 24px; 2px surface gap between touching bars
    left, right = label_w, 150  # right margin keeps tip labels inside the canvas
    plot_w = W - left - right
    top = 74 + (26 if legend else 0)
    group_h = [max(len(g.bars) * (bar_h + gap) - gap, 30) for g in groups]
    pitch = 22  # air between groups
    height = top + sum(group_h) + pitch * (len(groups) - 1) + 44

    def sx(v: float) -> float:
        return left + plot_w * v / xmax

    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" role="img" font-family="{FONT}">',
        f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>",
        f'<text x="0" y="24" font-size="18" font-weight="600" fill="{t["ink"]}">{escape(title)}</text>',
        f'<text x="0" y="48" font-size="13" fill="{t["ink2"]}">{escape(subtitle)}</text>',
    ]
    if legend:
        x = 0
        for name, key in legend:
            o.append(f'<rect x="{x}" y="66" width="12" height="12" rx="3" fill="{t[key]}"/>')
            o.append(f'<text x="{x + 18}" y="76.5" font-size="13" fill="{t["ink2"]}">{escape(name)}</text>')
            x += 18 + int(len(name) * 7.4) + 24

    bottom = height - 30
    for v in ticks:  # recessive hairline grid
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{bottom}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{bottom + 18}" font-size="12" fill="{t["muted"]}" '
                 f'text-anchor="middle" style="font-variant-numeric: tabular-nums">{tick_fmt(v)}</text>')
    o.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')

    y = top
    for g, gh in zip(groups, group_h, strict=True):
        mid = y + gh / 2
        name_y = mid + (-3 if g.note else 4.5)
        o.append(f'<text x="{left - 12}" y="{name_y:.1f}" font-size="13.5" fill="{t["ink"]}" '
                 f'text-anchor="end">{escape(g.name)}</text>')
        if g.note:
            o.append(f'<text x="{left - 12}" y="{mid + 13:.1f}" font-size="11.5" fill="{t["muted"]}" '
                     f'text-anchor="end">{escape(g.note)}</text>')
        by = y + (gh - (len(g.bars) * (bar_h + gap) - gap)) / 2
        for b in g.bars:
            w = max(sx(b.value) - left, 1.5)
            o.append(f'<path d="{_bar_path(left, by, w, bar_h)}" fill="{t[b.color]}"/>')
            o.append(f'<text x="{left + w + 8:.1f}" y="{by + bar_h / 2 + 4.5:.1f}" font-size="12.5" '
                     f'fill="{t["ink"]}" style="font-variant-numeric: tabular-nums">{escape(b.label)}</text>')
            by += bar_h + gap
        y += gh + pitch
    o.append("</svg>")
    return "\n".join(o)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%".replace(".0%", "%")


def kv_chart(theme: str) -> str:
    d = json.load(open(ART / "kv_bench.json"))["modes"]
    none = d["none"]["ttft_ms_mean"]
    rows = [
        ("No cache", "full prompt every time", "none", "neutral"),
        ("Prefix cache", "built once, kept in RAM", "prefix", "s1"),
        ("Persistent cache", "loaded from disk", "persistent", "s1"),
    ]
    groups = []
    for name, note, key, color in rows:
        m = d[key]
        label = f"{m['ttft_ms_mean']:,.0f} ms"
        if key != "none":
            label += f"  ·  {none / m['ttft_ms_mean']:.1f}× faster"
        groups.append(Group(name, [Bar(m["ttft_ms_mean"], color, label)], note))
    n_prefix = d["prefix"]["prefix_tokens"]
    new = d["prefix"]["prefilled_tokens_mean"]
    step = 250
    xmax = (int(none) // step + 1) * step
    return hbar_chart(
        title="Time to first token",
        subtitle=f"Mean of 20 requests · {n_prefix:,} cached prompt tokens, ~{new:.0f} new per request",
        groups=groups, xmax=xmax, ticks=list(range(0, xmax + 1, step)),
        tick_fmt=lambda v: f"{v:,.0f} ms" if v else "0", legend=[],
        desc="; ".join(f"{g.name}: {g.bars[0].label}" for g in groups), theme=theme,
    )


def accuracy_chart(theme: str) -> str:
    sets = [
        ("Dev set", "40 cases", "eval_mlx_r2.json"),
        ("Fresh test set", "24 cases, before tuning", "eval_fresh_before.json"),
        ("Fresh test set", "24 cases, final", "eval_fresh.json"),
        ("Test set", "60 cases, written before the LoRA data", "eval_test.json"),
    ]
    sets = [x for x in sets if (ART / x[2]).exists()]
    groups = []
    for name, note, f in sets:
        r = json.load(open(ART / f))
        groups.append(Group(name, [Bar(r["accuracy"], "s1", pct(r["accuracy"])),
                                   Bar(r["param_accuracy"], "s2", pct(r["param_accuracy"]))], note))
    return hbar_chart(
        title="Intent and parameter accuracy",
        subtitle="Qwen3-1.7B-4bit on MLX · default config (JSON grammar, strict validation)",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[("Intent accuracy", "s1"), ("Parameter accuracy", "s2")],
        desc="; ".join(f"{g.name} ({g.note}): intent {g.bars[0].label}, parameters {g.bars[1].label}"
                       for g in groups), theme=theme,
    )


def ablation_chart(theme: str) -> str:
    splits = [x for x in ("dev", "fresh", "test") if (ART / f"ablation_{x}.json").exists()]
    rows = {x: {(r["few_shot"], r["constraint"]): r
                for r in json.load(open(ART / f"ablation_{x}.json"))} for x in splits}
    names = {"dev": ("Dev (40)", "s1"), "fresh": ("Fresh (24)", "s2"), "test": ("Test (60)", "s3")}
    order = [
        ((False, "none"), "No examples, free"), ((False, "intent"), "No examples, intent mask"),
        ((False, "json"), "No examples, JSON grammar"), ((True, "none"), "Few-shot, free"),
        ((True, "intent"), "Few-shot, intent mask"), ((True, "json"), "Few-shot + JSON (R2)"),
    ]
    groups = []
    for key, label in order:
        bars = [Bar(rows[x][key]["param_acc"], names[x][1], pct(rows[x][key]["param_acc"]))
                for x in splits]
        fb = max(rows[x][key]["fallback"] for x in splits)
        groups.append(Group(label, bars, f"fallback up to {pct(fb)}"))
    return hbar_chart(
        title="What helps: few-shot examples and a JSON grammar",
        subtitle="Exact decisions (intent + parameters), strict validation in every row",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[names[x] for x in splits], label_w=210,
        desc="; ".join(f"{g.name}: " + ", ".join(b.label for b in g.bars) for g in groups),
        theme=theme,
    )


def kv_variants_chart(theme: str) -> str:
    e = json.load(open(ART / "kv_variants.json"))["e2e"]
    order = [("prefix/kv16", "16-bit KV (R2)"), ("prefix/kv8", "8-bit KV"), ("prefix/kv4", "4-bit KV"),
             ("none/kv16/win1024", "Window 1,024"), ("none/kv16/win512", "Window 512"),
             ("none/kv16/win256", "Window 256")]
    groups = []
    for key, name in order:
        r = e[key]
        note = (f"{r['kv_mb']:.0f} MB cached prefix" if key.startswith("prefix")
                else f"no cache, {r['ttft_ms_mean']:,.0f} ms TTFT")
        groups.append(Group(name, [Bar(r["exact_accuracy"], "s1", pct(r["exact_accuracy"]))], note))
    return hbar_chart(
        title="Smaller KV: 8-bit is free, 4-bit and short windows break routing",
        subtitle="Exact decisions on dev + fresh (64) · the prompt is 1.1k tokens long",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[], desc="; ".join(f"{g.name} ({g.note}): {g.bars[0].label}" for g in groups),
        theme=theme,
    )


def conversation_chart(theme: str) -> str:
    sm = json.load(open(ART / "conversation_bench.json"))["summary"]
    order = [("prefix/none", "No history"), ("prefix/sliding", "Sliding history (default)"),
             ("conversation/sliding", "Sliding + reuse"), ("prefix/block", "Block history"),
             ("conversation/block", "Block + reuse")]
    groups = []
    for key, name in order:
        a, b = sm[key]["turns_1_to_N"]["ttft_ms_mean"], sm[key]["after_window_full"]["ttft_ms_mean"]
        groups.append(Group(name, [Bar(a, "s1", f"{a:,.0f} ms"), Bar(b, "s2", f"{b:,.0f} ms")],
                            f"{pct(sm[key]['all']['exact_accuracy'])} exact"))
    top = max(max(b.value for b in g.bars) for g in groups)
    xmax = (int(top) // 100 + 1) * 100
    return hbar_chart(
        title="Conversation reuse only helps while the history prefix is stable",
        subtitle="Mean TTFT over 10 sessions × 10 turns · history ≤ 3 turns",
        groups=groups, xmax=xmax, ticks=list(range(0, xmax + 1, 100)),
        tick_fmt=lambda v: f"{v:,.0f} ms" if v else "0",
        legend=[("Turns 1–4", "s1"), ("Turns 5–10", "s2")],
        desc="; ".join(f"{g.name}: turns 1-4 {g.bars[0].label}, turns 5-10 {g.bars[1].label}"
                       for g in groups), theme=theme,
    )


def multistep_chart(theme: str) -> str:
    def reps(f):
        return {r["planner"]: r for r in json.load(open(ART / f))["reports"]}
    dev, fresh = reps("multistep_eval_dev.json"), reps("multistep_eval_fresh.json")
    order = [("clause", "Clause planner"), ("clause+", "Clause + and-split + guard (R2)"),
             ("model", "One-shot model planner")]
    groups = []
    for key, name in order:
        if key in dev and key in fresh:
            d, f = dev[key]["plan_accuracy"], fresh[key]["plan_accuracy"]
            groups.append(Group(name, [Bar(d, "s1", pct(d)), Bar(f, "s2", pct(f))],
                                f"refuses {pct(fresh[key]['reject_recall'])} of must-refuse"))
    return hbar_chart(
        title="Multi-step plans: exact-plan accuracy",
        subtitle="Plans are validated, never executed · fresh set written before the planner options",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[("Dev (61)", "s1"), ("Fresh (30)", "s2")], label_w=240,
        desc="; ".join(f"{g.name}: dev {g.bars[0].label}, fresh {g.bars[1].label}" for g in groups),
        theme=theme,
    )


def datasets_chart(theme: str) -> str:
    """Share of each intent in every evaluation set (and the synthetic training set)."""
    from collections import Counter

    data = ROOT / "data"
    sets = [("eval_heldout.jsonl", "Dev (40)", "s1"), ("eval_fresh.jsonl", "Fresh (24)", "s2"),
            ("eval_test.jsonl", "Test (60)", "s3")]
    counts = {}
    for f, label, _ in sets:
        rows = [json.loads(x) for x in (data / f).read_text().splitlines() if x.strip()]
        counts[label] = (Counter(r["expected_intent"] for r in rows), len(rows))
    synth = data / "synth" / "manifest.json"
    syn = json.load(open(synth)) if synth.exists() else None
    intents = sorted({i for c, _ in counts.values() for i in c},
                     key=lambda i: -sum(c[i] / n for c, n in counts.values()))
    groups = []
    for i in intents:
        bars = [Bar(counts[label][0][i] / counts[label][1], key,
                    f"{counts[label][0][i]}") for _, label, key in sets]
        note = f"{syn['by_intent'].get(i, 0)} synthetic" if syn else ""
        groups.append(Group(i.replace("_", " "), bars, note))
    sub = "Share of each set · bar label = number of cases"
    if syn:
        sub += f" · synthetic LoRA set: {syn['train']} train / {syn['valid']} valid"
    return hbar_chart(
        title="What the evaluation sets contain",
        subtitle=sub, groups=groups, xmax=0.35, ticks=[0, 0.1, 0.2, 0.3],
        tick_fmt=lambda v: f"{v:.0%}", legend=[(lbl, key) for _, lbl, key in sets],
        desc="; ".join(f"{g.name}: " + ", ".join(b.label for b in g.bars) for g in groups),
        theme=theme, label_w=190,
    )


def lora_chart(theme: str) -> str:
    r = json.load(open(ART / "lora_compare.json"))["results"]
    sets = [("dev", "Dev (40)"), ("fresh", "Fresh (24)"), ("test", "Test (60)")]

    def pooled(key):
        return (sum(sum(r[key][s]["ok"]) for s, _ in sets)
                / sum(r[key][s]["n"] for s, _ in sets))

    rows = [(name, [r[k][s]["exact"] for k in ("base/none", "base/json", "adapter/json")])
            for s, name in sets]
    rows.append(("All 124", [pooled(k) for k in ("base/none", "base/json", "adapter/json")]))
    groups = [Group(name, [Bar(v, key, pct(v)) for v, key in zip(vals, ("s1", "s2", "s3"),
                                                                    strict=True)])
              for name, vals in rows]
    return hbar_chart(
        title="LoRA adapter vs base model: exact accuracy",
        subtitle="0.15% of parameters trained on 700 synthetic examples · paired McNemar p = 0.013",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[("Base, free decoding", "s1"), ("Base, JSON grammar", "s2"),
                ("LoRA adapter", "s3")], label_w=120,
        desc="; ".join(f"{g.name}: " + ", ".join(b.label for b in g.bars) for g in groups),
        theme=theme,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts = [("kv-ttft", kv_chart), ("accuracy", accuracy_chart), ("ablation", ablation_chart),
              ("kv-variants", kv_variants_chart), ("conversation", conversation_chart),
              ("multistep", multistep_chart), ("datasets", datasets_chart),
              ("lora", lora_chart)]
    for name, fn in charts:
        for theme in THEMES:
            (OUT / f"{name}-{theme}.svg").write_text(fn(theme), encoding="utf-8")
            print(f"wrote docs/images/charts/{name}-{theme}.svg")


if __name__ == "__main__":
    main()

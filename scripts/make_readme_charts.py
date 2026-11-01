#!/usr/bin/env python3
"""Draw the README charts from the measured results in artifacts/.

Writes light and dark SVGs to docs/images/charts/ (the README swaps them with <picture>).
Re-run after `training/bench_kv.py` or `training/evaluate.py` to refresh the figures:

    uv run python scripts/make_readme_charts.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "images" / "charts"

# Validated categorical pair (blue, orange) + neutral, per mode.
THEMES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#6f6e69", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "neutral": "#a3a29b",
    },
    "dark": {
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#9a9993", "grid": "#2c2c2a",
        "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "neutral": "#6b6a64",
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
    ]
    groups = []
    for name, note, f in sets:
        r = json.load(open(ART / f))
        groups.append(Group(name, [Bar(r["accuracy"], "s1", pct(r["accuracy"])),
                                   Bar(r["param_accuracy"], "s2", pct(r["param_accuracy"]))], note))
    return hbar_chart(
        title="Intent and parameter accuracy",
        subtitle="Qwen3-1.7B-4bit on MLX · 0% malformed JSON in every run",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[("Intent accuracy", "s1"), ("Parameter accuracy", "s2")],
        desc="; ".join(f"{g.name} ({g.note}): intent {g.bars[0].label}, parameters {g.bars[1].label}"
                       for g in groups), theme=theme,
    )


def ablation_chart(theme: str) -> str:
    rows = {(r["few_shot"], r["constrained"]): r for r in csv.DictReader(open(ART / "ablation.csv"))}
    order = [
        (("False", "False"), "Plain prompt"),
        (("False", "True"), "Constrained decoding only"),
        (("True", "False"), "Few-shot examples only"),
        (("True", "True"), "Both (R2 default)"),
    ]
    groups = []
    for key, name in order:
        r = rows[key]
        ia, pa, bad = float(r["intent_acc"]), float(r["param_acc"]), float(r["malformed"])
        groups.append(Group(name, [Bar(ia, "s1", pct(ia)), Bar(pa, "s2", pct(pa))],
                            f"{pct(bad)} malformed JSON"))
    return hbar_chart(
        title="What helps: few-shot examples vs constrained decoding",
        subtitle="Dev set, 40 cases · each technique switched on and off",
        groups=groups, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], tick_fmt=lambda v: f"{v:.0%}",
        legend=[("Intent accuracy", "s1"), ("Parameter accuracy", "s2")],
        desc="; ".join(f"{g.name}: intent {g.bars[0].label}, parameters {g.bars[1].label}, {g.note}"
                       for g in groups), theme=theme,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("kv-ttft", kv_chart), ("accuracy", accuracy_chart), ("ablation", ablation_chart)]:
        for theme in THEMES:
            (OUT / f"{name}-{theme}.svg").write_text(fn(theme), encoding="utf-8")
            print(f"wrote docs/images/charts/{name}-{theme}.svg")


if __name__ == "__main__":
    main()

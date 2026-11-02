#!/usr/bin/env python3
"""Quantized KV (kv_bits 8/4) and rotating-window (max_kv_size) study.

    HF_HUB_OFFLINE=1 uv run python training/bench_kv_variants.py [--part micro|e2e|all]

micro: for prefix lengths n (R2's system prompt, repeated), compares a full-precision
       prefix cache with 8- and 4-bit copies of it. Measures KV bytes, the cached-suffix
       forward (TTFT proxy), one decode step, and how far the last-position logits move
       (max |Δ|, argmax, top-5 overlap, KL).
e2e:   routes the dev (40) and fresh (24) requests through R2's normal path with
       kv_bits = 0/8/4 (prefix cache) and with rotating windows (cache_mode none,
       max_kv_size = 256/512/1024 vs unbounded). Reports accuracy, agreement with the
       full-precision run, TTFT and peak memory.

Uses the locally cached model only. Writes artifacts/kv_variants.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import system_prompt_for  # noqa: E402
from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.models.hub import local_or_repo  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

LENGTHS = [1118, 2048, 4096, 8192]
BITS = [0, 8, 4]
WINDOWS = [0, 1024, 512, 256]  # 0 = unbounded
STEP = 2048


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000


def micro(args) -> list[dict]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    mx.set_cache_limit(256 * 2**20)
    settings = load_settings()
    model, tok = load(local_or_repo(settings.model.model_id))
    base = tok.encode(system_prompt_for(settings, default_registry()))
    lengths = [n for n in LENGTHS if n <= args.max_len]
    while len(base) < max(lengths):
        base = base + base
    req = tok.apply_chat_template(
        [{"role": "user", "content": "find the swift files in this project please"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + '{"intent": "'
    suffix = tok.encode(req, add_special_tokens=False)[:22]
    s = len(suffix)

    def forward(tokens, cache):
        arr = mx.array(tokens)
        out = None
        for i in range(0, len(tokens), STEP):
            out = model(arr[i: i + STEP][None], cache=cache)
        return out[:, -1, :].astype(mx.float32)

    for n in (128, 2048):  # compile kernels
        c = make_prompt_cache(model)
        mx.eval(forward(base[:n], c))

    rows = []
    for n in lengths:
        full = make_prompt_cache(model)
        mx.eval(forward(base[:n], full))
        ref = None
        for bits in BITS:
            cache = copy.deepcopy(full)
            if bits:
                cache = [c.to_quantized(group_size=64, bits=bits) for c in cache]
            mx.eval([c.state for c in cache])
            kv_mb = sum(c.nbytes for c in cache) / 1e6
            times, logits = [], None
            for _ in range(args.repeats):
                with Timer() as t:
                    logits = forward(suffix, cache)
                    mx.eval(logits)
                times.append(t.ms)
                trim_prompt_cache(cache, s)
            # one decode step with an (n + s)-token cache
            forward(suffix, cache)
            dec = []
            for _ in range(args.repeats):
                with Timer() as t:
                    mx.eval(forward(suffix[-1:], cache))
                dec.append(t.ms)
                trim_prompt_cache(cache, 1)
            row = {"n": n, "kv_bits": bits or 16, "kv_mb": round(kv_mb, 1),
                   "suffix_ms": round(statistics.median(times), 1),
                   "decode_ms": round(statistics.median(dec), 2)}
            if ref is None:
                ref = logits
            else:
                p = mx.softmax(ref, axis=-1)
                q = mx.softmax(logits, axis=-1)
                kl = mx.sum(p * (mx.log(p + 1e-12) - mx.log(q + 1e-12)))
                top_ref = set(mx.argsort(-ref[0])[:5].tolist())
                top_q = set(mx.argsort(-logits[0])[:5].tolist())
                row.update(
                    max_dlogit=round(float(mx.max(mx.abs(ref - logits))), 3),
                    same_argmax=int(mx.argmax(ref).item()) == int(mx.argmax(logits).item()),
                    top5_overlap=len(top_ref & top_q),
                    kl=float(kl),
                )
            rows.append(row)
            print(row)
            del cache
        del full
        mx.clear_cache()
    return rows


def e2e(args) -> dict:
    settings = load_settings()
    settings.model.backend = "mlx"
    settings.model.cache_mode = "prefix"
    backend = create_backend(settings)
    runner = EvaluationRunner(settings, backend)
    cases = load_eval_cases(ROOT / "data" / "eval_heldout.jsonl") + load_eval_cases(
        ROOT / "data" / "eval_fresh.jsonl")

    configs = [("prefix", b, 0) for b in BITS] + [("none", 0, w) for w in WINDOWS]
    out = {}
    for mode, bits, window in configs:
        key = f"{mode}/kv{bits or 16}" + (f"/win{window}" if mode == "none" else "")
        backend.kv_bits, backend.max_kv_size = bits, window or None
        settings.model.cache_mode = mode
        backend.set_cache_mode(mode)
        if mode != "none":
            backend.warmup(runner.engine.system_prompt)
        runner.engine.decide("what can you do")  # compile this configuration's kernels
        rep = runner.run_cases(cases)
        rows = rep.cases
        out[key] = {
            "n": len(rows),
            "intent_accuracy": round(sum(r.intent_ok for r in rows) / len(rows), 4),
            "exact_accuracy": round(sum(r.params_ok for r in rows) / len(rows), 4),
            "fallback_rate": round(rep.fallback_rate, 4),
            "ttft_ms_mean": round(statistics.fmean(r.metrics.get("ttft_ms", 0) for r in rows), 1),
            "peak_memory_gb": max(r.metrics.get("peak_memory_gb", 0) for r in rows),
            "kv_mb": round(backend.cache_stats().extra.get("nbytes", 0) / 1e6, 1),
            "decisions": [(r.predicted, r.parameters) for r in rows],
            "ok": [r.params_ok for r in rows],
        }
        print(key, {k: v for k, v in out[key].items() if k not in {"decisions", "ok"}})
    for ref_key in ("prefix/kv16", "none/kv16/win0"):
        ref = out[ref_key]["decisions"]
        for key, row in out.items():
            if key.split("/")[0] == ref_key.split("/")[0]:
                row["same_decisions_vs_" + ref_key] = sum(
                    a == b for a, b in zip(ref, row["decisions"], strict=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["micro", "e2e", "all"], default="all")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kv_variants.json")
    args = ap.parse_args()

    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.part in ("micro", "all"):
        result["micro"] = micro(args)
    if args.part in ("e2e", "all"):
        result["e2e"] = e2e(args)

    if "micro" in result:
        print("\n| n | KV bits | KV MB | Suffix fwd | Decode step | max Δlogit | argmax "
              "| top-5 | KL |")
        print("|---:|---:|---:|---:|---:|---:|---|---:|---:|")
        for r in result["micro"]:
            print(f"| {r['n']} | {r['kv_bits']} | {r['kv_mb']} | {r['suffix_ms']} ms | "
                  f"{r['decode_ms']} ms | {r.get('max_dlogit', '–')} | "
                  f"{'same' if r.get('same_argmax', True) else 'DIFF'} | "
                  f"{r.get('top5_overlap', 5)}/5 | {r.get('kl', 0):.2e} |")
    if "e2e" in result:
        print("\n| Config | Intent | Exact | Same as full-precision | TTFT | Peak mem | KV MB |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for key, r in result["e2e"].items():
            same = next((v for k, v in r.items() if k.startswith("same_decisions_vs_")), "")
            print(f"| {key} | {r['intent_accuracy']:.1%} | {r['exact_accuracy']:.1%} | "
                  f"{same}/{r['n']} | {r['ttft_ms_mean']} ms | {r['peak_memory_gb']:.2f} GB | "
                  f"{r['kv_mb']} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

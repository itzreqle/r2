#!/usr/bin/env python3
"""How prefill, cached-suffix prefill, decode, KV memory and disk I/O scale with context length.

Measures, for prefix lengths n (tokens of R2's real system prompt, repeated as needed):

* uncached TTFT proxy  - one forward pass over n + s tokens from an empty cache
* cached   TTFT proxy  - forward over only the s suffix tokens with an n-token prefix cache
* decode step          - one token with an (n + s)-token cache (memory-bandwidth bound)
* KV bytes / offset    - what the cache actually allocates
* save / load          - safetensors round-trip of the prefix cache
* equivalence          - max |logit difference| of cached vs uncached at the last position

The forward pass ends at the logits of the last position (what TTFT waits for); sampling,
tokenization and HTTP are excluded, so these are lower bounds on end-to-end TTFT.

    uv run python training/bench_scaling.py [--repeats 5] [--suffix 22]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import system_prompt_for  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.models.hub import local_or_repo  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

LENGTHS = [128, 256, 512, 1024, 1118, 1536, 2048, 3072, 4096, 6144, 8192]
STEP = 2048  # prefill chunk, same as R2 and mlx-lm


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--suffix", type=int, default=22, help="new tokens per request (R2 mean)")
    ap.add_argument("--max-len", type=int, default=4096, help="largest prefix (memory grows)")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kv_scaling.json")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import (
        load_prompt_cache,
        make_prompt_cache,
        save_prompt_cache,
        trim_prompt_cache,
    )

    # Keep the benchmark polite on a 16 GB machine: small buffer cache, freed between steps.
    mx.set_cache_limit(256 * 2**20)
    lengths = [n for n in LENGTHS if n <= args.max_len]

    settings = load_settings()
    model_id = settings.model.model_id
    model, tok = load(local_or_repo(model_id))

    # Real token streams: R2's system prompt for the prefix, a real request for the suffix.
    sys_text = system_prompt_for(settings, default_registry())
    base = tok.encode(sys_text)
    while len(base) < max(lengths):
        base = base + base
    req = tok.apply_chat_template(
        [{"role": "user", "content": "find the swift files in this project please"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + '{"intent": "'
    suffix = tok.encode(req, add_special_tokens=False)
    suffix = (suffix * 4)[: args.suffix]
    s = len(suffix)

    def prefill(tokens: list[int], cache) -> mx.array:
        arr = mx.array(tokens)
        out = None
        for i in range(0, len(tokens), STEP):
            out = model(arr[i : i + STEP][None], cache=cache)
        return out[:, -1, :]

    class Timer:
        """`with Timer() as t: ...` then t.ms (wall clock, after mx.eval inside the block)."""

        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self.ms = (time.perf_counter() - self.t0) * 1000

    # Warm up Metal kernels for every shape class we will hit.
    for n in (128, 2048):
        c = make_prompt_cache(model)
        mx.eval(prefill(base[:n], c))

    rows = []
    for n in lengths:
        prefix = base[:n]
        full = prefix + suffix

        unc = []
        for _ in range(args.repeats):
            c = make_prompt_cache(model)
            with Timer() as t:
                last_unc = prefill(full, c)
                mx.eval(last_unc)
            unc.append(t.ms)

        cache = make_prompt_cache(model)
        with Timer() as t:
            mx.eval(prefill(prefix, cache))
        build_ms = t.ms
        kv_bytes = int(sum(x.nbytes for x in cache))
        offset = cache[0].offset

        cached = []
        for _ in range(args.repeats):
            with Timer() as t:
                last_cached = prefill(suffix, cache)
                mx.eval(last_cached)
            cached.append(t.ms)
            trim_prompt_cache(cache, s)

        diff = mx.abs(last_cached.astype(mx.float32) - last_unc.astype(mx.float32))
        max_dlogit = float(mx.max(diff))
        same_argmax = int(mx.argmax(last_cached).item()) == int(mx.argmax(last_unc).item())

        # decode: one token on top of n + s cached tokens
        mx.eval(prefill(suffix, cache))
        dec = []
        one = mx.array([suffix[0]])[None]
        for _ in range(args.repeats * 2):
            with Timer() as t:
                mx.eval(model(one, cache=cache))
            dec.append(t.ms)
            trim_prompt_cache(cache, 1)
        trim_prompt_cache(cache, s)

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "c.safetensors")
            with Timer() as t:
                save_prompt_cache(path, cache)
            save_ms = t.ms
            file_bytes = Path(path).stat().st_size
            loads = []
            for _ in range(args.repeats):
                with Timer() as t:
                    lc = load_prompt_cache(path)
                    mx.eval([x.state for x in lc])
                loads.append(t.ms)

        row = {
            "prefix_tokens": n,
            "suffix_tokens": s,
            "uncached_ms": unc,
            "cached_ms": cached,
            "decode_ms": dec,
            "uncached_ms_median": statistics.median(unc),
            "cached_ms_median": statistics.median(cached),
            "decode_ms_median": statistics.median(dec),
            "speedup": statistics.median(unc) / statistics.median(cached),
            "build_ms": build_ms,
            "kv_bytes": kv_bytes,
            "kv_offset": offset,
            "save_ms": save_ms,
            "load_ms_median": statistics.median(loads),
            "file_bytes": file_bytes,
            "max_abs_logit_diff": max_dlogit,
            "same_argmax": same_argmax,
        }
        rows.append(row)
        del cache
        mx.clear_cache()
        print(
            f"n={n:5d}  uncached {row['uncached_ms_median']:7.1f} ms"
            f"  cached {row['cached_ms_median']:6.1f} ms"
            f"  x{row['speedup']:5.1f}  decode {row['decode_ms_median']:5.1f} ms"
            f"  kv {kv_bytes / 2**20:6.1f} MiB  load {row['load_ms_median']:5.1f} ms"
            f"  |dlogit| {max_dlogit:.3g} argmax_same={same_argmax}",
            flush=True,
        )

    cfg = json.load(open(Path(local_or_repo(model_id)) / "config.json"))
    report = {
        "model_id": model_id,
        "model_config": {k: cfg.get(k) for k in (
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "hidden_size", "vocab_size", "quantization")},
        "hardware": {
            "chip": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30,
            "macos": platform.mac_ver()[0],
        },
        "software": {p: _version(p) for p in ("mlx", "mlx-lm")},
        "repeats": args.repeats,
        "prefill_step": STEP,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def _version(pkg: str) -> str:
    import importlib.metadata as md

    return md.version(pkg)


if __name__ == "__main__":
    raise SystemExit(main())

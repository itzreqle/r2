"""Same-model baseline: R2's prompt on Ollama (llama.cpp), warm vs. after an unload.

Uses the 20 dev utterances of ``bench_kv.py``, R2's exact rendered prompt (same tokenizer and
chat template) and, by default, R2's decision schema as Ollama's ``format``. Two phases:

* ``restart``: the model is unloaded before every request, as after a process restart.
  Ollama reports model load and prompt evaluation separately, so we can report both.
* ``warm``:    the model stays loaded; Ollama reuses the longest prefix of the previous
  prompt that is still in its KV state (here the 1.1k-token system prefix).

Requires a running Ollama with the model already pulled; never pulls.
Usage: uv run python training/bench_ollama.py [--model qwen3:1.7b] [--constraint json|none]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import IntentEngine  # noqa: E402
from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.models.ollama_backend import OllamaBackend  # noqa: E402
from narrow_ai.models.prompt_templates import INTENT_PREFILL  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402


def summarise(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows]
    return {"mean": st.mean(v), "sd": st.pstdev(v), "median": st.median(v)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--constraint", choices=["json", "none"], default="json")
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_heldout.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "artifacts" / "ollama" / f"bench_{args.constraint}.json"

    settings = load_settings()
    settings.model.constraint = "json"  # build the schema; unused when --constraint none
    backend = OllamaBackend(args.model, max_tokens=settings.model.max_tokens)
    engine = IntentEngine(settings, backend, default_registry(settings))
    schema = engine.schema if args.constraint == "json" else None
    utts = [c.utterance for c in load_eval_cases(args.eval)][: args.limit]

    def ask(u: str) -> dict:
        msgs = [{"role": "system", "content": engine.system_prompt}, {"role": "user", "content": u}]
        r = backend.generate(msgs, prefill=INTENT_PREFILL, constraint_schema=schema)
        return {
            "utterance": u,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.generate_time_ms,
            "load_ms": r.load_time_ms,
            "prompt_eval_ms": (r.meta or {}).get("prompt_eval_ms", 0.0),
            "ttft_minus_load_ms": r.ttft_ms - r.load_time_ms,
            "text": r.text,
        }

    restart = []
    for u in utts:
        backend.unload()
        time.sleep(0.5)
        restart.append(ask(u))
    ask(utts[0])  # make sure the system prefix is resident
    warm = [ask(u) for u in utts]

    keys = ("ttft_ms", "total_ms", "load_ms", "prompt_eval_ms", "ttft_minus_load_ms")
    report = {
        "model": args.model,
        "constraint": args.constraint,
        "n": len(utts),
        "restart": {k: summarise(restart, k) for k in keys},
        "warm": {k: summarise(warm, k) for k in keys},
        "same_outputs": sum(a["text"] == b["text"] for a, b in zip(restart, warm, strict=True)),
        "cases": {"restart": restart, "warm": warm},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for phase in ("restart", "warm"):
        s = report[phase]
        print(
            f"{phase:8s} ttft={s['ttft_ms']['mean']:.0f}±{s['ttft_ms']['sd']:.0f} ms "
            f"load={s['load_ms']['mean']:.0f} prompt_eval={s['prompt_eval_ms']['mean']:.0f} "
            f"ttft-load={s['ttft_minus_load_ms']['mean']:.0f} total={s['total_ms']['mean']:.0f}"
        )
    print(f"identical outputs restart vs warm: {report['same_outputs']}/{len(utts)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

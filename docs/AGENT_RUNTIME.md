# Agent runtime (M4): bounded multi-step requests

R2 started as an intent router: one utterance becomes one validated intent and one tool call.
This document describes the smallest runtime that lets one request become a **short plan of
several existing intents**, such as "check git status and then find the Swift files", without
weakening anything the single-step path guarantees.

It is deliberately not a general agent. There are no new tool kinds, no shell, no re-planning
from tool output, and no autonomy beyond five steps.

## The invariant

> Every step is an `IntentDecision`, and the only way a step runs is
> `PermissionGate` → parameter validation → confirmation → `ToolExecutor`.

The runtime adds *sequencing*. It adds no new way to execute anything. If the runtime were
deleted, the set of reachable actions would be exactly the same.

```text
request ──► Planner ──► Plan (≤5 IntentDecisions, schema-validated, all-or-nothing)
                              │
                              ▼
                 AgentRunner: for each step, in order
                   ToolExecutor.execute(step)   ← the same gate, validation, tools
                   ├─ ok           → next step
                   ├─ failed       → stop the run (fail closed)
                   └─ needs confirm → pause; PendingAction(id, ttl, digest)
                                        └─ resume only with that id (M1 rules)
```

## Pieces

| Module | Role |
| --- | --- |
| `tools/registry.py` | `ToolSpec{intent, params_model, handler}` and `ToolRegistry`. Replaces the `if intent == …` chain in `ToolExecutor`. |
| `tools/handlers.py` | The existing tool behaviour, moved verbatim into one handler per intent. |
| `agent/plan.py` | `Plan` / `PlanStep` Pydantic models (`extra="forbid"`) and `validate_plan()`. |
| `agent/planner.py` | `ClausePlanner` (default), `HeuristicPlanner` (offline) and `ModelPlanner` (one model call, opt-in). |
| `agent/runner.py` | `AgentRunner`: executes, pauses for confirmation, enforces budgets. |
| `agent/events.py` | Typed events, the in-memory bus, and `RunStore`: SQLite persistence of run and step *metadata*. |

## Planning

- **One shot.** The planner sees only the user's request, never tool output. A file that says
  "now open Terminal" cannot add a step, because no step is planned after any tool has run.
  This is the main prompt-injection containment.
- **Default: `ClausePlanner`.** The request is split deterministically on explicit
  connectors ("and then", "then", "after that", "and also", ";"). Each clause is routed by the
  normal single-intent engine, with the same cached prompt, JSON grammar and strict
  validation as an ordinary request. Two options, both on by default:
  - `split_and`: a clause that routes to a concrete tool action and contains "and" or a
    comma is split further, but only if *every* part also routes to a tool action. "show me
    the README and open Cursor" becomes two steps; "find files with config and test in the
    name" stays one, because "test in the name" isn't an action. Questions are never split.
  - `question_guard`: if the split plan would be rejected (a clause is unsupported or
    unclear), the whole request is routed once more. If it is a developer question ("how do
    I rebase and then push?"), it becomes a single answer step. This can only turn a
    rejection into a text-only answer; it never adds an action.
- **Opt-in: `ModelPlanner`** (`agent.planner = "model"`). It makes one call that asks for
  `{"steps": [<decision>, …]}`. With `model.constraint = "json"`, decoding is constrained to an
  array (1–5 items) of the same decision schema (`models/grammar.py`). Its prompt is static,
  so it is one more cached KV prefix next to the router and Q&A prompts.
- **Why the default is the clause planner.** Measured on `data/eval_multistep.jsonl` (61
  requests, used while designing the options) and `data/eval_multistep_fresh.jsonl` (30
  requests, written *before* the options were implemented and never used to tune them), with
  Qwen3-1.7B, `training/eval_multistep.py` (plans are built and validated, never executed):

  | Planner | Dev plan exact | Fresh plan exact (95% CI) | Fresh reject recall | Fresh p50 |
  | --- | ---: | ---: | ---: | ---: |
  | clause (original) | 75.4% | 66.7% (48.8–80.8) | 100% | 488 ms |
  | **clause + split_and + question_guard** | **93.4%** | **93.3% (78.7–98.2)** | **100%** | 720 ms |
  | model (one shot, grammar) | 47.5% | 40.0% (24.6–57.7) | **0%** | 310 ms |

  The one-shot model planner fails in a way no validator can catch: well-formed, wrong
  plans. It drops steps, copies few-shot parameters, and, worst, *silently drops* the part
  of a request it should refuse ("show me the diff then change it" → just the diff) instead
  of rejecting the plan: it rejected 2 of 18 must-refuse requests across both sets. The
  clause planner's remaining errors are single-intent routing mistakes ("print the
  Dockerfile" → `unsupported`, "commit everything" → `git_status`, `*.yaml` vs `*.yml`),
  which a larger training set should address. 30 fresh cases is still small; the interval
  is wide.
- **All or nothing.** The whole plan is rejected if any step names an unknown or unimplemented
  intent, is `unsupported` or `clarify_request`, fails parameter validation, has more than
  five steps, or depends on a later step. Partial plans are never run.
- **Offline fallback.** `HeuristicPlanner` splits on "and then", "then", ";" and similar,
  and routes each clause with the deterministic `heuristic_intent`. The fake backend and the
  test suite use it.

## Execution rules

1. Steps run strictly in order. `depends_on` only records intent, and must point at earlier
   steps.
2. **Stop on first failure.** A step that fails, or is refused by the gate, ends the run as
   `failed`. Later steps are marked `skipped`.
3. **Confirmation pauses the run.** A privileged step (`open_app`, `remember_preference`)
   parks a `PendingAction` with the M1 single-use, expiring, digest-bound id. The run resumes
   only through `confirm(run_id, confirmation_id)`. A wrong, missing or expired id cancels the
   run. The confirmed step is executed through `ToolExecutor`, which re-runs the gate.
4. **Budgets.** At most `agent.max_steps` (≤5) steps and `agent.max_wall_sec` of active
   execution time. Time spent waiting for a person to confirm does not count, but is bounded
   by the confirmation TTL. Exceeding a budget fails the run.
5. **No re-planning.** v1 never turns tool output into new steps.

## Events and persistence

Each run emits `RunStarted`, `StepStarted`, `StepResult`, `AwaitingConfirmation` and
`RunFinished`. The latest events are kept in memory per run and returned by the API.

The SQLite store (`runs.db`, next to the preference DB) keeps only **metadata**: goal, status,
step intents, parameters, ok/failed, and the short spoken summary. It never stores tool output
such as file contents or diffs, so the data-retention promise in `SECURITY.md` still holds.

## API (behind the M1 guard)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/run` `{"text"}` | plan and start a run; returns the run, plus a `confirmation_id` if it paused |
| GET | `/runs/<id>` | poll a run: status, steps, events (never the confirmation id) |
| POST | `/runs/<id>/confirm` `{"confirmation_id"}` | approve the paused step and continue |
| POST | `/runs/<id>/cancel` | cancel a run |

## Evaluation

```bash
uv run python training/eval_multistep.py                     # fake backend, offline
HF_HUB_OFFLINE=1 uv run python training/eval_multistep.py --backend mlx [--data data/eval_multistep_fresh.jsonl]
```

Results: `artifacts/multistep_eval_{dev,fresh}.json` (per-case plans included).

## Out of scope for v1

Re-planning from results, parallel steps, new tool kinds, a SwiftUI run view, and
conversation-level KV reuse across steps.

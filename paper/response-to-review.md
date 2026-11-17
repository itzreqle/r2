# Response to review — revision 3

This note maps each review point to what changed in `r2-whitepaper.tex` and lists what still
needs new experiments. Section numbers refer to revision 3.

| # | Comment (summary) | Status |
|---|---|---|
| 1 | Too long; reads like a technical report; repetitive; informal tone | **Done**: new 8-page IEEE version `r2-ieee.tex` |
| 2 | Related work lacks explicit, quantitative comparison (Prompt Cache, SGLang, llama.cpp) | **Done**, including a measured same-model llama.cpp baseline (§7.2) |
| 3 | Theory is mostly basic properties of causal models; focus on fingerprinting and layer validation | **Done** |
| 4 | Too much implementation detail in Architecture/Implementation; move to appendix | **Done** |
| 5 | Evaluation: few samples, one device, no baseline, exact match only | **Partly done** (soft metrics, llama.cpp baseline); larger sets and more hardware **pending** |
| 6 | Novelty overstated; Proposition 1 is not new; revise contributions | **Done** |
| 7 | Limitations: data are self-made; synthetic training data | **Done** |

## Two versions of the paper

- **`r2-ieee.tex` / `r2-ieee.pdf` (new, for IEEE and arXiv):** IEEE journal template (`IEEEtran`).
  The main text ends on page 6, and main text plus references fit in **8 pages**: 7 full pages plus
  a few references on page 8, where the appendices begin. That is within IEEE IoTJ's 8-page
  threshold, which counts references. The main text has 10 floats instead of 25:
  - Three tables are merged into one accuracy table (Table V: base, Ollama and LoRA, with CI,
    macro-F1, graded score and McNemar *p*).
  - A new two-panel figure (Fig. 4, `figures/comparison.pdf`) shows R2 vs llama.cpp latency and the
    outcome of every error.
  - Everything else moved to Appendices A–F. No concept was dropped: implementation, constrained
    decoding proof, requirements table, all secondary figures and tables, LoRA analysis, prompt,
    data and reproducibility are all still there.
  - Build with `./build_ieee.sh`. It regenerates `references-ieee.bib`, a copy of `references.bib`
    without the "Peer-reviewed" notes.
- **`r2-whitepaper.tex` (extended version):** the full-length text, kept as a technical-report
  companion. It also restores the conversation-reuse figure and the requirements table, which an
  earlier revision step had dropped.

## What changed

**1 and 4: length, tone and structure.** The main text is reorganised as Introduction, Related
Work, Problem, System Design, Analysis, Setup, Results, Discussion and Limitations, and Conclusion.
The two code listings, the request algorithm, the file-tool hardening, the HTTP API, the clients,
software versions, the formal constrained-decoding argument and the LoRA rank and subspace analysis
moved to Appendices A–C. The former Architecture and Implementation sections are now one
"System Design" section built around four design decisions (D1–D4). The informal section title
"Aren't Existing Solutions Good Enough?", the "new in revision 2" markers and the revision history
were removed, and repeated statements were merged (for example, the 72 ms / 82 ms figures and the
"few-shot examples drive accuracy" finding now each appear once in Results and once in the abstract
and conclusion).

**2: positioning.** Added Section 2.1, "Positioning for a single-user laptop", and an extended Table 1
that now includes CachedAttention (USENIX ATC '24), LMCache and Ollama next to vLLM, SGLang,
Prompt Cache, CacheBlend and llama.cpp. It states three concrete differences and why each matters on
a laptop:
(i) survival across restarts, quantified as an 82 ms validated load against a 1.52 s rebuild;
(ii) validity: every other system decides reuse by token match only, whereas a KV cache also depends
on weights, tokenizer, template and adapter;
(iii) an execution policy.
It also cites the published speed-ups of Prompt Cache, CachedAttention and SGLang, and explains why
those numbers are not directly comparable with ours.
*Honest framing:* persistence itself is not new (CachedAttention, LMCache and llama.cpp slot files
all save KV state), so the claim now rests on the validity check, not on persistence.

**3 and 6: theory and contributions.** Prefix invariance is now a one-paragraph *Remark* marked as
well known, with no proof. The new core of Section 5 is Section 5.2, "Validity of a persisted cache".
It contains a definition of configuration validity, the fingerprint formula, and Proposition 1
("fail-closed reuse") with a proof. It also states the guarantee's limits explicitly: the
size-and-mtime identity used for large files, the 80-bit collision bound, no payload checksum, and
runtime versioning. The memory and cost model is condensed to one subsection. The contribution list
now says that the value is the integration of existing techniques plus validated persistence and a
safety layer, and the abstract states that R2 introduces no new inference technique.

**5: evaluation, the part done now.** Added graded and outcome-based metrics (Table 9, Section 7.5):
intent macro-F1, a graded parameter score (partial credit, in the spirit of SLURP's SLU-F1), and an
outcome taxonomy covering *correct*, *near miss*, *safe stop*, *wrong read-only*, *wrong gated* and
*missed refusal*. The table also reports *useful* and *no-unintended-change* rates. These are computed
by the new script `training/soft_metrics.py` from the stored predictions (`artifacts/soft_metrics.json`),
so no model was rerun. Finding: 9 of the base model's 14 errors are safe stops, and with the adapter
none are.

**7: limitations.** Section 8.1 now has separate paragraphs for author-constructed data, synthetic
training data (including why word-overlap decontamination does not remove shared phrasing templates),
sample size (citing Bowyer et al., ICML 2025, on interval estimates below a few hundred datapoints),
one machine and one runtime, metric proxies, and the scope of the guarantees.

**5: llama.cpp baseline (done, §7.2, Table 5).** Same Qwen3-1.7B on Ollama 0.40.13 (Q4_K_M GGUF),
same rendered prompt, R2's schema as Ollama's `format`, and R2's own parser, validator and grader, via the
new `OllamaBackend` (`src/narrow_ai/models/ollama_backend.py`, unit-tested offline).
- Warm prefix: 72 ms (R2) vs 79 ms (Ollama), so no real difference.
- After an unload or restart: Ollama re-evaluates the prompt in 1,878 ms, while R2 loads its validated
  prefix in 82 ms (23× less).
- Accuracy: 110/124 on both with the grammar, with 6 vs 6 discordant requests (McNemar p = 1.0).
  Without the grammar it's 107 vs 104 (p = 0.58).
- The paper now says plainly that R2's advantage is restart cost, validity checking and policy, not
  warm latency or accuracy.
- Raw outputs: `artifacts/ollama/`; scripts: `training/bench_ollama.py`, `training/evaluate.py --backend ollama`.

## Pending experiments for point 5 (need your decision or other hardware)

These cannot honestly be written into the paper until they have been run.

**5a. Larger, less biased evaluation sets (target at least 400 single-intent requests).**
At 90% accuracy, n = 400 gives a Wilson half-width of about ±3 points, against ±5–12 now.
- *Independent writers:* 5–10 developers who have not seen the prompt each write about 30–40
  requests from a one-page description of the 11 intents. This also addresses point 7.
- *External out-of-scope requests:* about 100 utterances sampled from a public spoken-language
  corpus (SLURP or MASSIVE), all labelled `unsupported`. These test refusal on language we did not
  write. This requires downloading a dataset.
- Keep the current 124 as they are and report the old and new sets separately.

**5b. Other hardware.** Run `bench_kv.py`, `bench_scaling.py` and `evaluate.py` on one or two other
Apple silicon Macs, ideally one base M1 or M2 with 8 GB and one M-series Pro or Max. The scripts are
self-contained. The cost model predicts how the timings scale with throughput and bandwidth, so the
fitted coefficients on each machine are themselves a result.

**5c. Optional user study.** 5–10 participants rate R2's response to each of their own requests
(useful / acceptable / wrong), to complement the automatic outcome metrics.

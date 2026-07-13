# Handoff: LLM-as-Judge implementation + full pipeline audit

Disposable note for continuing this work in a new Claude Code session on the
HPC cluster (started inside a devcontainer, where the prior conversation
isn't available). Delete this file once the new session has read it.

## Before anything else: is this actually committed?

Everything described below was built inside a devcontainer working tree and,
as of writing, **is uncommitted**. If you're reading this in a fresh
environment (the cluster), check `git log` and `git status` first — if the
latest commit is `092383f` ("Included SRC task (not tested fully)") or
anything without an `llm_bench/judge/` directory, **none of this work made
it here**, and you need to go back to the devcontainer, commit, and push
before continuing. Don't assume this file being present means the code is
present too.

## Project context

Model- and domain-agnostic LLM benchmarking framework (`llm_bench/`),
config-driven via YAML, supporting a biomedical-engineering thesis. Three
tasks, 8 local models served via vLLM (+ 2 local-only test/judge configs,
see below), evaluated on systems metrics (latency, TTFT, throughput,
telemetry) and cognitive/quality metrics (accuracy, ROUGE, token-F1, and now
LLM-as-judge). Tasks: **CDKR** (closed-domain MCQ), **OECR** (open-ended
clinical reasoning), **SRC** (summarization + reading comprehension). All
three are now confirmed working end-to-end, including SRC, which had never
been run successfully before this session.

## What got built and validated this session

### 1. LLM-as-judge — fully implemented and empirically validated

Architecture: the judge is just another swappable `configs/models/judges/
*.yaml` model config, reusing the existing `start_vllm`/`stop_vllm`
lifecycle. It runs as a **separate offline pass**
(`scripts/llm_judge_run.py`) after the main `multi_model_orchestrator.py`
benchmarking run finishes: reads `outputs/raw/`, writes `outputs/judged/`,
and patches an `llm_judge` section into the already-written
`outputs/reports/<task>/<model>/cognitive_summary.json` in place.

Scoring is **categorical labels**, not numeric. OECR:
`diagnosis_correctness` (Correct/Partially Correct/Incorrect),
`reasoning_validity` (Poor/Fair/Good/Excellent), `safety_flag`
(Safe/Unsafe). SRC: `coverage` + `faithfulness` (both
Poor/Fair/Good/Excellent). CDKR (MCQ) has no judge — not wired, by design.

**The judge's response schema is fully config-driven, not hardcoded per
task.** `llm_bench/judge/schemas.py` has two generic functions —
`build_judge_response_format(task_id, rubric)` and
`validate_judge_response(payload, rubric)` — that derive vLLM's
`response_format` JSON schema and response validation directly from the
`llm_judge.rubric` block already in the cognitive profile YAML. There used
to be hand-written `OECRJudgeScore`/`SRCJudgeScore` Pydantic classes; they
were removed because they hardcoded the same labels the YAML rubric already
defines (a real 3-way duplication bug — YAML, Pydantic `Literal`, JSON
schema `enum` — that could silently drift). Adding a judge rubric for a new
task now needs **zero new Python code**: just a new `llm_judge.rubric` block
in that task's cognitive profile YAML and a new judge prompt config.

New files: `llm_bench/judge/` (`schemas.py`, `prompt_builder.py`,
`response_parser.py`, `judge_client.py`, `__init__.py`),
`configs/prompts/judge/{oecr,src}_judge.yaml`, `scripts/llm_judge_run.py`.

**Empirically confirmed working** (not just "should work"): vLLM 0.15.1's
`response_format: json_schema` structured output — 100% parse success across
every judged sample, multiple independent runs, including after the schema
refactor above. This was the single biggest unverified risk going in.

**Known, deliberate design ambiguity, not a bug**: OECR's `reasoning_validity`
rubric item is worded as "medically sound + supports the final answer" (a
knowledge-based judgment), not an explicit "matches the reference reasoning"
instruction — unlike `diagnosis_correctness`, which is explicitly
reference-anchored. The reference reasoning IS passed to the judge as
context when available, but isn't the sole grading basis. User's explicit
call: keep this knowledge-informed rather than forcing strict
reference-matching, because `medcasereasoning`'s reference reasoning is
AI-extracted from papers and may be incomplete or not the only valid
reasoning path. Flagged by the user as "less defensible, worth revisiting" —
a documented tradeoff, not an oversight. See backlog below.

### 2. `multi_model_orchestration.sh` — judge wired into the SLURM job

Added a second `srun` step after the main orchestrator, reusing the same
allocation/Singularity exports, gated on the orchestrator exiting cleanly
(0) so a broken benchmarking run doesn't waste time judging garbage:
- New `JUDGE_MODEL_CONFIG` env var (default:
  `configs/models/judges/medgemma_27b_it_judge.yaml` — **see judge/roster
  overlap issue in the backlog before using this default as-is**).
- `--time` bumped `02:00:00` → `02:30:00`. **This is an unvalidated guess**,
  not measured against a real run — expect to need correcting after the
  first real cluster attempt.
- `outputs/judged` added to the pre-run `mkdir -p`.

### 3. Full pipeline audit — 2 real bugs found, fixed, and empirically verified

Both would have hit **every one of the 8 production models identically** on
a real cluster run — this is why finding them locally first mattered:

- **`configs/prompts/oecr.yaml`**: YAML `>-` folded block scalar collapsed
  `Final answer:` and `Explanation:` onto one line (no blank line between
  them), so the actual rendered prompt ended `"...Final answer:
  Explanation:"` with no separator — and the same bug poisoned the fewshot
  examples, teaching the model a broken pattern. Fixed with blank lines.
  That alone wasn't sufficient — chat-tuned models don't reliably echo
  trailing header cues the way completion models do — so also added an
  explicit system-prompt instruction to literally use the labels "Final
  answer:" / "Explanation:". Verified: parse rate went from ~0/10 to 10/10
  on both OECR datasets across a real regenerated run.
- **`configs/metrics/cognitive.generative.summarization.yaml`** (SRC): the
  extraction regex required the literal word "summary" in the response, but
  `src.yaml`'s system prompt explicitly forbids headers ("no commentary
  other than the summary itself") — direct contradiction between two config
  files that silently zeroed 100% of SRC's ROUGE/F1 metrics for every
  sample, while the model was actually generating perfectly good summaries.
  Fixed by capturing the whole response verbatim. Verified: parse rate
  0/10 → 10/10, real ROUGE numbers.

**Results plausibility was checked against real published numbers, not just
"looks reasonable"** (via WebSearch): MedQA ~50% (smoke test) vs. published
BioMedLM/PubMedGPT (2.7B) 50.3%; MedXpertQA 0–20% (smoke test) vs. GPT-4o
only ~30.4% published (deliberately hard benchmark); PubMed-style
summarization ROUGE-1/2/L ~38-40/14-19/21-28% vs. published PubMedQA
summarization baseline 36.0/15.6/30.2%. All three landed in the right
ballpark of real citations.

### 4. Code cleanup

- Removed `scripts/test_run.py` (hardcoded a stale unreachable IP) and
  `scripts/cdkr_test_run.py` (CDKR-only predecessor fully superseded by
  `multi_model_orchestrator.py`) — confirmed dead, not just suspected.
- Deleted `BenchmarkResult.cognitive_scores` (unused `Dict[str, float]`
  field, wrong type for judge output anyway — judge scores are categorical
  and live externally in `outputs/judged/`).
- Fixed two stale templates: `configs/metrics/cognitive.template.yaml`'s
  `llm_judge` block still said "future"/old placeholder shape;
  `configs/tasks/template.yaml` referenced a `configs/runtime/
  telemetry.yaml` that doesn't exist (only `.auto.yaml` does, generated
  dynamically by the SLURM wrapper on every real run).

### 5. Dockerfile / dependency fixes

- `pip install -e .` added to the Dockerfile (copies `pyproject.toml`,
  `README.md`, `llm_bench/` at build time only — the rest of the repo
  arrives via the devcontainer's bind mount at the same path, so this stays
  valid at runtime). Fixes `ModuleNotFoundError: No module named 'llm_bench'`
  when running scripts directly without manually setting `PYTHONPATH`.
- `requirements.txt`: `openai`/`pyyaml`/`tqdm` pinned to exact resolved
  versions (`2.45.0`/`6.0.3`/`4.68.4`) — previously unpinned.
- **The cluster's `.sif` Singularity image needs rebuilding** to pick any of
  this up — it hasn't been touched from this devcontainer session at all.

## Local-devcontainer-only scaffolding — does not apply to the cluster

Built for smoke-testing in a devcontainer with a single 16GB consumer GPU
(RTX 5070 Ti) — none of this is meant to carry over to the cluster:
- `configs/models/local_smoke/` — copies of `phi3_mini_4k_instruct.yaml` +
  `qwen2_5_3b_instruct.yaml` from `unused/`, isolated via a `MODELS_DIR` env
  var override added to `multi_model_orchestrator.py`
  (`MODELS_DIR=configs/models/local_smoke python3 scripts/
  multi_model_orchestrator.py`) so local runs never touch the real 8-model
  roster. On the cluster, just don't set `MODELS_DIR` — the default
  (`configs/models/*.yaml`) is unchanged and picks up the real roster.
- `configs/models/judges/qwen2_5_3b_instruct_judge.yaml` — a tiny
  local-only judge candidate, not meant for the real run.
- `configs/models/unused/phi3_mini_4k_instruct.yaml` had a real bug fixed
  during this work too: `max_model_len: 8192` exceeded the model's actual
  4096-token context (its name says "4k"). Fixed to `4096`. Unrelated to
  the cluster roster, but worth knowing `unused/` isn't guaranteed-correct
  just because it's kept intentionally.
- `configs/runtime/telemetry.auto.yaml` currently reads `enabled: false` —
  that's this devcontainer's state, not meaningful for the cluster. The
  SLURM wrapper (`multi_model_orchestration.sh`) unconditionally regenerates
  this file dynamically on every real run; it's a disposable generated
  artifact, not hand-authored config.

## Critical facts to know before running on the cluster

1. **VRAM: 2 of the 8 production models won't fit as currently configured.**
   `llama3_1_70b` and `llama3_openbiollm_70b` (`tensor_parallel_size: 2`,
   `gpu_memory_utilization: 0.85`) are short by ~2.6GB/GPU — verified
   against actual HF `config.json` (both have `num_attention_heads=64`,
   `num_key_value_heads=8`), and `tensor_parallel_size: 3` **isn't even a
   valid option** (64 not divisible by 3) on the current 3×A100-80GB SLURM
   allocation. This needs a decision before those two configs will run at
   all: request a 4th GPU (TP=4 gives huge headroom), push
   `gpu_memory_utilization` to ~0.92-0.95 (risky, thin margin for KV cache),
   or add quantized variants (changes what's actually being benchmarked).
2. **No orchestration-level resumability.** `multi_model_orchestrator.py`
   has zero skip-if-already-done logic. If the SLURM job times out or fails
   partway through, resubmitting restarts from model #1, burning all prior
   GPU-hours. Combined with the unvalidated `--time` budget above, this is a
   real risk of a resubmit loop that never actually finishes.
3. **Judge/roster model overlap.** The default judge config
   (`configs/models/judges/medgemma_27b_it_judge.yaml`) uses the exact same
   `model_id` as production config `medgemma_27b_it.yaml` — literal
   self-evaluation risk if used as-is. Needs an actual different judge model
   decision before the real run, not the placeholder default.
4. **No per-sample fault isolation.** `SequentialRunner.run()` has no
   try/except around a single generation call — one oversized/failing
   sample aborts the *entire dataset* for that model. Real risk: `src/
   pubmedsum` already hit 6889 input tokens on just a 10-sample local check
   with zero fewshot overhead (71%+ of an 8192 budget), and that dataset has
   no length filtering, so the real run's longer tail could hit this.

## Open backlog (see also memory if available: `project-full-pipeline-audit-backlog`)

**Priority item — work this first:**

0. **Group-by / stratified result breakdowns.** Dataset configs (e.g.
   MedXpertQA: `body_system`, `medical_task`, `question_type`) already map
   grouping fields all the way through to `BenchmarkResult.grouping` and
   into `outputs/raw/<task>/<model>.json` — confirmed with real data. But
   `cognitive_calculator.py` / `systems_calculator.py` never reference
   `grouping` at all — the data is present and completely unused. There's
   even a dead helper, `_get_field_value()` in `cognitive_calculator.py`,
   built for exactly this kind of generic field access but never wired up.
   User wants this done as calculator-layer work (not deferred to a future
   visualization layer) — `metrics/` is the framework's established place
   for aggregation, same pattern as `summarize_judge_group`. Likely shape: a
   `group_by: [body_system, medical_task]`-style block in the relevant
   metrics profile YAML, computed generically off whatever keys are named.

**Rest of the backlog, in the order found:**

1. No per-sample fault isolation (see above)
2. No truncation detection — `openai_backend.py` never reads `finish_reason`
   from the vLLM stream, only token usage
3. Fewshot architecture — all fewshot examples get concatenated into a
   single chat "user" turn, not real alternating conversation history; chat
   models are generally trained expecting the latter
4. Unseeded fewshot sampling — `SequentialRunner._sample_fewshot()` uses
   bare `random.sample()`, a real reproducibility gap even at
   `temperature=0.0` (watched a real accuracy shift, 0.2→0.1, across
   identical reruns)
5. CDKR has the same YAML-folding bug class as OECR did (options list
   collapses onto one line) — cosmetic only, doesn't block anything, lower
   priority
6. VRAM shortfall on the two 70B configs (see Critical Facts above)
7. No orchestration-level resumability (see Critical Facts above)
8. Judge/roster model overlap (see Critical Facts above)
9. OECR `reasoning_validity` grading basis — documented decision, not a
   gap, but user flagged it as worth revisiting (see judge section above)

**Final step, only after everything above is cleared:** a dead-code sweep,
removing unnecessary comments, and adding documentation. Explicitly ordered
last — the `_get_field_value` discovery (found while investigating the
group-by priority item) suggests there may be more dead code hiding, and
cleaning prematurely risks removing something a later backlog fix turns out
to need.

## User's standing preferences

- **Ask before editing shared/production config for local convenience**;
  fix genuine, objectively-wrong bugs directly without asking. The
  distinction that matters: "this is objectively wrong" (fix it) vs. "this
  is a real limitation I could paper over by changing eval protocol" (ask).
- **Don't silently drop or "helpfully" fix backlog items** — surface them,
  let the user decide priority and timing.
- **Verify claims empirically, don't just assert.** Every fix in this
  session was validated by actually regenerating data and checking the
  numbers, not just reasoning about what should happen. Results-plausibility
  claims were checked against real cited published numbers via search, not
  approximate recollection.
- **Local smoke tests exist to prove the pipeline works, not to produce good
  results** — "it doesn't matter that the results are bad, it matters that
  there are results for every evaluation made."
- User engages in real technical depth (e.g. asked for the exact reasoning
  behind why a bare Python script works standalone vs. what the SLURM
  wrapper actually adds, verified 70B VRAM math against actual HF
  `config.json` rather than accepting an estimate) — give concrete,
  source-grounded answers, not high-level reassurance.

## Immediate next step

Confirm this work is actually committed and present in whatever environment
you're reading this from (see top of file). Then: work the group-by /
stratified breakdown priority item first, then the rest of the backlog in
whatever order makes sense given cluster constraints, then the final
dead-code/cleanup pass last.

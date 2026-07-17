# Handoff: first real Deucalion test run

Disposable note for continuing this work in a new Claude Code session on the
Deucalion cluster (this was written from inside a devcontainer, where the
prior conversation isn't available to you). Delete this file once you've
read it and don't need it anymore — it's a snapshot, not documentation.

## 0. Before anything else: is this actually committed?

**As of writing, it is not.** The devcontainer's last commit is `b6be9bc`
("Implemented orchestration-level resumability... SLURM wrapper
consolidation + Dead-code pass and bug fixes on SLURM scripts"). Everything
below this line describes work sitting **uncommitted** on top of that
commit. If you're reading this on the cluster and `git log` shows `b6be9bc`
as `HEAD` with a clean `git status`, **none of this has arrived yet** — go
back to the devcontainer, commit, push, then pull here before doing
anything else. Don't assume this file's presence means the code is present.

Uncommitted changes as of writing (`git status --short` in the
devcontainer):
```
 D configs/models/judges/medgemma_27b_it_judge.yaml
 D configs/models/judges/qwen2_5_3b_instruct_judge.yaml
 D configs/models/llama3_8b_instruct.yaml
 D configs/models/mistral_7b_instruct_v03.yaml
 D configs/models/phi3_mini_4k_instruct.yaml
 M configs/models/qwen2_5_3b_instruct.yaml
 D configs/models/qwen3_32b_awq.yaml
 M configs/runs/local_smoke.yaml
 M scripts/orchestration/deucalion_orchestrator.sh
 M scripts/orchestration/llm_judge_run.py
 M scripts/orchestration/orchestrator.py
 M scripts/orchestration/run_pipeline.sh
 M scripts/orchestration/slurm_orchestrator.sh
?? configs/models/granite_3_1_2b_instruct.yaml
?? configs/models/judges/deepseek_v4_flash_judge.yaml
?? configs/models/judges/gemma3_4b_it_judge.yaml
?? configs/models/judges/gpt_oss_120b_judge.yaml
```
This includes real functional fixes (`RUN_CONFIG` now required, per-cluster
`WORKDIR`, the judge model pick, a `--time` budget increase) — running the
cluster job against the old committed state would hit already-fixed bugs.

## 1. What you're about to do

Run the **first-ever real test** of this project's full production pipeline
on actual SLURM hardware. Nothing in the changes below has touched a real
cluster before — everything so far was verified inside a devcontainer with
no SLURM available, via `bash -n` syntax checks and mocked
`sbatch`/`scontrol`/`srun`/`singularity` calls. This run is explicitly a
**short, complete test**, not the definitive full-scale benchmark: every
dataset is capped at `eval_limit: 10` samples (already set in
`configs/tasks/{cdkr,oecr,src}.yaml`), but it still exercises the *entire*
roster — all 8 production models, all 3 tasks, all 6 enabled datasets, plus
the judge pass. Nothing else needs changing before running it.

**Command** (run with `bash`, not `sbatch` — the wrapper submits the job
itself):
```
bash scripts/orchestration/deucalion_orchestrator.sh
```
That's it — no env vars need setting for the default path.
`RUN_CONFIG` defaults to `configs/runs/full_production.yaml` (all 8 models,
all 3 tasks) and `JUDGE_MODEL_CONFIG` defaults to
`configs/models/judges/deepseek_v4_flash_judge.yaml`, both already wired
into the wrapper.

**Before running**: double-check your shell doesn't have a leftover
`RUN_CONFIG`/`JUDGE_MODEL_CONFIG`/`WORKDIR` exported from earlier
experimentation — `echo $RUN_CONFIG $JUDGE_MODEL_CONFIG $WORKDIR` should
print nothing, or exactly what you intend, since the wrapper only applies
its defaults when those variables are unset.

## 2. Known risks specific to this run — read before submitting

- **`deepseek_v4_flash_judge.yaml` has never been booted anywhere.** It's a
  284B-total/13B-active MoE model (`deepseek-ai/DeepSeek-V4-Flash`,
  released April 2026) picked for being unrelated in family to every
  production model in the roster. VRAM math (~156GB weight footprint
  against 4x80GB A100 at TP=4) suggests it should fit, but this has only
  ever been checked on paper — this run is the actual test. **If it fails
  to boot**, don't debug it mid-run: the orchestrator's own benchmarking
  pass is unaffected either way (the judge pass is a separate `srun` step
  that only starts after the orchestrator exits cleanly). Re-run just the
  judge pass afterward with the prepared fallback:
  ```
  JUDGE_MODEL_CONFIG=configs/models/judges/gpt_oss_120b_judge.yaml \
    python3 -u scripts/orchestration/llm_judge_run.py \
    --judge-model configs/models/judges/gpt_oss_120b_judge.yaml
  ```
  (`gpt-oss-120b`, TP=1, designed by OpenAI to fit a single 80GB GPU — the
  well-known, lower-risk option.) This reads whatever's already in
  `outputs/raw/`, so it doesn't require re-running the benchmarked models.

- **`--time=06:00:00` is a reasoned estimate, not a measured one.** Worst
  case: 8 production models × up to `startup_timeout_s: 2400` (40 min) each
  to boot ≈ 5.3 hours, before any generation time or the judge pass. Bumped
  from `02:30:00` specifically because that budget couldn't plausibly cover
  8 large-model boots. If this run finishes well under 6 hours, that's
  useful real data — note the actual wall-clock so a future run's `--time`
  can be calibrated from measurement instead of worst-case arithmetic.

- **Resumability protects against collateral damage within a run, not
  against redoing work across a resubmit.** If this job fails partway
  through (e.g. at model #5 of 8), the per-dataset fault isolation added
  recently means model #1-4's results won't be *destroyed* by whatever
  killed model #5 — but if you resubmit the whole job, `orchestrator.py`
  still restarts from model #1 every time. There is no auto-skip of
  already-completed models on a fresh invocation (explicitly deferred,
  tracked as a known gap). A mid-run failure still costs the GPU-hours
  already spent on earlier models if you have to resubmit.

- **Two of the 8 models are 70B and use `tensor_parallel_size: 4`**
  (`llama3_1_70b`, `llama3_openbiollm_70b`) — this assumes a real 4-GPU
  allocation, matching `deucalion_orchestrator.sh`'s `--gpus=4`. If the
  actual allocation ever comes back different, these two will fail
  (64 attention heads isn't divisible by anything other than 1/2/4/8/16/32/64,
  so a 3-GPU allocation specifically would break TP validity for these).

## 3. What to check after the run

- `outputs/reports/run_summary.json` — top-level pass/fail per task.
- `outputs/reports/<task_id>/summary.json` — per-model entries now include a
  `datasets` field with `{attempted, succeeded, error}` per dataset (added
  this week specifically for this kind of granular post-run diagnosis).
- `outputs/reports/<task_id>/<model>/cognitive_summary.json` — for OECR/SRC,
  confirm the `llm_judge` block got merged in (proves the judge pass
  actually ran and parsed correctly, not just that it didn't crash).
- Actual wall-clock time for the whole job, and ideally a breakdown of how
  long each model took to boot — this is the first real data point for
  calibrating `--time` and `startup_timeout_s` values that have been guesses
  until now.
- Whether `deepseek_v4_flash_judge.yaml` booted successfully or needed the
  `gpt_oss_120b_judge.yaml` fallback (see above).

## 4. Project context, briefly

Model- and domain-agnostic LLM benchmarking framework (`llm_bench/` — the
installable package) with SLURM/Singularity orchestration scaffolding
(`scripts/`), built for a biomedical-engineering thesis. Three tasks:
**CDKR** (closed-domain MCQ), **OECR** (open-ended clinical reasoning),
**SRC** (summarization/reading comprehension). 8 production models served
via vLLM, evaluated on systems metrics (latency, TTFT, throughput,
telemetry) and cognitive metrics (accuracy, ROUGE, token-F1, plus an
offline LLM-as-judge pass for OECR/SRC — CDKR has no judge, by design).
`orchestrator.py` runs the benchmarking pass; `llm_judge_run.py` runs the
separate judge pass afterward, reading `outputs/raw/` and patching
`outputs/judged/` + `cognitive_summary.json` in place.

Config structure: `configs/tasks/*.yaml` define what to run (fixed, flat,
non-recursive discovery — `TASKS_DIR`/`MODELS_DIR` in `orchestrator.py`).
`configs/runs/*.yaml` (`{"tasks": [...], "models": [...]}`) select a subset
by file stem — `$RUN_CONFIG` names which one, and is now *required* (raises
loudly if unset, no more silent "discover everything" fallback). Everything
else referenced from inside a task config (prompt, dataset, metrics profile
paths) is an unconstrained path string, not anchored to `configs/`.

## 5. Backlog — condensed, self-contained (don't assume memory access here)

**Architectural decisions, yours to call:**
- OECR's `reasoning_validity` judge rubric grades on medical soundness, not
  strict reference-matching — a deliberate, previously-made call, kept open
  because it was flagged as "worth revisiting."
- Reports-per-run vs. overwrite-in-place (`outputs/reports/` currently
  overwrites every invocation) — actively being decided, not just noted.

**Potential improvements, low priority, only after everything else:**
- Fewshot examples are concatenated into one flat chat turn instead of real
  multi-turn message history — a real interface change, not urgent.
- API-hosted models (Gemini specifically, for its free tier) — verified
  Gemini's OpenAI-compatible endpoint supports everything the existing
  `OpenAIBackend`/`JudgeClient` classes need; the actual blocker is that
  `orchestrator.py`/`llm_judge_run.py` unconditionally boot a local vLLM
  server for every model, with no path to just point at an external API.
- LLM-jury (3-judge majority vote instead of one judge) — architecturally
  cheap to add (the aggregation layer doesn't care how many judges fed into
  it), but 3x the judge-pass compute cost is the real blocker, plus an
  unresolved tie-break rule for genuine 3-way label disagreement.

**Known upcoming work, not yet scoped:** the user has flagged "a major task
regarding output analysis" they intend to work before concluding this
backlog, to be detailed in a future session. No further detail exists yet —
don't guess at its shape, just be aware it's coming.

**Final step, deferred until everything above is done:** a dead-code sweep,
now scoped to just the Python side (`llm_bench/`, `scripts/orchestration/*.py`)
since the SLURM `.sh` scripts already got a dead-code pass.

## 6. Standing preferences (carried forward, still apply)

- Ask before editing shared/production config for local convenience; fix
  genuine, objectively-wrong bugs directly without asking.
- Don't silently drop or "helpfully" resolve backlog items — surface them.
- Verify claims empirically. Every fix in this project's history was
  validated by actually running something and checking real output, not
  just reasoning about what should happen.
- Never commit without being explicitly asked, even when a commit would
  obviously be useful (see section 0).

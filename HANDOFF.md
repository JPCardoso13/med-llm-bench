# Handoff: pre-flight fixes for the first real Deucalion test run

Disposable note for continuing this work in a different Claude Code session
(devcontainer or otherwise) - this was written from a session running
directly on the Deucalion cluster, where the prior conversation isn't
available to you. Delete this file once you've read it and don't need it
anymore - it's a snapshot, not documentation.

## 0. Before anything else: is this actually committed?

**As of writing, it is not.** `git log` on Deucalion shows `HEAD` at
`67577fd` ("SLURM orchestration script bug fixes + Added definitive
LLM-as-a-Judge choice + Fixed local smoke test pipeline bugs + Final run
readiness check") with everything below sitting **uncommitted on top of
it, on the Deucalion filesystem only**. Unlike the last handoff, this one
did *not* originate from a devcontainer-commit-push-pull flow - it was
written live on the cluster, so there is nothing to `git pull` yet. If
you're reading this anywhere other than that same Deucalion checkout, the
changes below don't exist in your filesystem at all; this file is the only
record of them until someone commits and pushes from Deucalion. Don't
assume git history reflects any of this.

`git status --short` on Deucalion as of writing:
```
 M scripts/asset_caching/download_hf_assets.py
 M scripts/asset_caching/download_hf_assets.sh
 M scripts/orchestration/deucalion_orchestrator.sh
 D scripts/vllm/python_vllm.sh
?? scripts/vllm/python_vllm_deucalion.sh
?? scripts/vllm/python_vllm_pipeline.sh
?? scripts/vllm/python_vllm_slurm.sh
```

Suggested commit message, in this repo's established style (short clauses
joined by " + "), if/when someone commits this:
```
Fixed HF_HOME cache-path bug across SLURM/asset scripts + Repurposed
asset-caching script for judge downloads and access checks + Split
python_vllm.sh into per-cluster wrappers + shared pipeline body + Bumped
deucalion time budget for judge boot time
```

## 1. What prompted this session

Picking up right where the last handoff left off: about to run the
first-ever real test of the full production pipeline on actual SLURM
hardware (`bash scripts/orchestration/deucalion_orchestrator.sh` - 8
models, 3 tasks, 6 datasets, `eval_limit: 10`, plus the judge pass). Before
firing that off, this session did a pre-flight review the previous session
hadn't done: read through the orchestration scripts for staleness, and
specifically checked the Hugging Face cache path against what's actually
on disk (the user pointed out the real cache lives one directory *above*
the project dir, not inside it).

## 2. Bugs found and fixed this session

**Real, objectively-wrong bug (fixed directly, matching standing
preference to fix genuine bugs without asking):**

- `run_pipeline.sh` (shared job body, untouched) defaults `HF_HOME` to
  `$WORKDIR/.cache/huggingface` - a directory that does not exist. The
  actual populated cache is at `$(dirname $WORKDIR)/.cache/huggingface`
  (i.e. `/projects/F202500001HPCVLABEPICURE/jcardoso/.cache/huggingface`,
  a sibling of the project dir, confirmed on disk - 52GB for
  `gemma-3-27b-it` alone, real weights not empty dirs). Under
  `HF_OFFLINE=1` (Deucalion's default), a wrong `HF_HOME` means every
  model load fails outright with no download fallback - this would have
  killed the very first real run before it started.
- Fixed by exporting the correct `HF_HOME` explicitly in
  **`deucalion_orchestrator.sh`** (cluster-specific override, same pattern
  already used there for `WORKDIR`/`HF_OFFLINE`/etc - `run_pipeline.sh`
  itself was left alone since its generic fallback is only wrong on this
  one cluster).
- The identical wrong-default bug also existed in
  `scripts/asset_caching/download_hf_assets.sh` - fixed the same way.

**Verified clean, no changes needed:** `orchestrator.py`,
`llm_judge_run.py`, `vllm_manager.py`, `run_pipeline.sh` - no obsolete or
dead code paths on the critical run path. Confirmed `med-llm-bench.sif`
exists, `singularity` is on `PATH`, SLURM account is valid, all 8
production models in `configs/runs/full_production.yaml` are cached and
populated, and all 6 enabled datasets are available (5 via the HF cache,
`multiclinsum` from local `data/processed/multiclinsum/*.jsonl`, no HF
dependency at all).

**Judge model: none of the 3 candidates were cached.** Deucalion has no
general internet access, so `deepseek_v4_flash_judge.yaml`,
`gpt_oss_120b_judge.yaml`, and `gemma3_4b_it_judge.yaml` (`DeepSeek-V4-Flash`,
`gpt-oss-120b`, `gemma-3-4b-it` respectively) were all unreachable as-is.
**Decision made this session: only download `DeepSeek-V4-Flash`** (the
original first-choice judge from the last handoff), to save time/bandwidth
rather than fetching all 3 candidates. This is an operational choice for
*this run*, not a resolution of the architectural judge-selection question
still open in the backlog below.

**`download_hf_assets.py`/`.sh` repurposed to actually fetch it.** The
script predates `configs/models/judges/` (confirmed via `git log`: script
last touched in `2811060`, `judges/` subdir introduced two commits later in
`aa65cfd`) and never scanned that directory, which is exactly why the judge
models were never cached in the first place. Changes:
- Default model sweep now also walks `configs/models/judges/*.yaml`.
- Added `--models PATH [PATH ...]` to target specific config files instead
  of the full sweep.
- Added `--skip-models` / `--skip-datasets`.
- Added `--check-only`: a fast, metadata-only `HfApi().model_info()` call
  per targeted repo (no download) that reports `OK`/`DENIED` per repo -
  built so gated-repo access can be verified in seconds instead of
  discovering a license-acceptance problem partway through a multi-hour
  download. Reusable for any future model, not just this one.
- `download_hf_assets.sh` now forwards `"$@"` to the Python script
  (previously took no arguments at all).

**`scripts/vllm/python_vllm.sh` (a separate, standalone manual launcher for
ad-hoc Python/vLLM scripts under SLURM - not part of the orchestrator path)
was flagged by the user as inconsistent and split, mirroring the existing
`run_pipeline.sh`/`deucalion_orchestrator.sh`/`slurm_orchestrator.sh`
pattern:**
- It had `#SBATCH --partition=rtx4060 --account=haslab` pragmas (small
  cluster) but a hardcoded Deucalion `WORKDIR` and `HF_OFFLINE=1` default -
  internally contradictory, and as written could not actually run
  correctly on either cluster.
- Split into `python_vllm_pipeline.sh` (shared job body, `HF_HOME` bug
  fixed the same way, `HF_OFFLINE` last-resort default flipped to `0` to
  match `run_pipeline.sh`'s convention), `python_vllm_deucalion.sh`
  (correct Deucalion partition/account/`WORKDIR`), and
  `python_vllm_slurm.sh` (correct rtx4060/haslab partition/account/
  `WORKDIR` - named to match the existing `slurm_orchestrator.sh`
  convention for that cluster, not "haslab").
- Old `scripts/vllm/python_vllm.sh` removed (`git rm`, uncommitted).
- **Usage change**: invocation goes from `sbatch scripts/vllm/python_vllm.sh
  <script> [args]` to `bash scripts/vllm/python_vllm_deucalion.sh <script>
  [args]` (or `_slurm.sh` on the small cluster) - matching how every other
  wrapper in this repo is launched. Resource shape (`--nodes=2`, no
  explicit `--gpus`, `--time=02:00:00`) was kept identical to the original
  on both wrappers rather than guessed at per-cluster; revisit if that
  turns out to be wrong for either cluster's actual needs.

**Time budget bumped.** `deucalion_orchestrator.sh`'s `--time` raised from
`06:00:00` to `08:00:00`. The original 6h estimate (from the last handoff)
only accounted for `8 models x up to 2400s boot each ~= 5.3h` and never
added the judge's own boot time. Now that the judge is pinned to
`DeepSeek-V4-Flash` (`startup_timeout_s: 3600`), worst-case boot time alone
is `5.3h + 1h ~= 6.3h`, already past the old budget before counting any
actual generation or judge-scoring time.

## 3. Current run status - read this first

**A SLURM job is already in flight and unfinished.** Job `1760803`
(`download_hf_assets`, submitted via `sbatch
scripts/asset_caching/download_hf_assets.sh --models
configs/models/judges/deepseek_v4_flash_judge.yaml --skip-datasets
--check-only`) was submitted to verify HF access to `DeepSeek-V4-Flash`
before committing to the real ~156GB download. As of writing it has been
**`PD` (pending, reason: `Priority`) for over 24 hours** - the
`normal-a100-80` partition is under heavy contention (75 jobs pending at
last check). SLURM's own backfill estimate currently projects a start
around `2026-07-22T16:22:29` (re-check with `squeue -j 1760803 --start`,
since this estimate has already slipped once and may again). This is
**not a stuck or broken job** - just a long queue. No output/error log
exists yet at `logs/asset_download/out/download_hf_assets_1760803.out` /
`logs/asset_download/err/download_hf_assets_1760803.err` because it
hasn't started running.

**The planned follow-up chain, agreed with the user but not yet
executed:**
1. Job `1760803` finishes -> read its log. If `DENIED`, stop and surface
   it rather than guessing at a fix.
2. If `OK` -> submit the real download, scoped to just
   `DeepSeek-V4-Flash`: `sbatch scripts/asset_caching/download_hf_assets.sh
   --models configs/models/judges/deepseek_v4_flash_judge.yaml
   --skip-datasets` (no `--check-only`).
3. If that download completes cleanly -> do a final readiness pass (model
   landed under the correct `HF_HOME`, no stray
   `RUN_CONFIG`/`JUDGE_MODEL_CONFIG`/`WORKDIR` env leakage, SIF/account
   still fine - same checklist the original handoff laid out) and only
   then run `bash scripts/orchestration/deucalion_orchestrator.sh`.
4. Any failure at any step -> stop and report back rather than retrying
   blindly or improvising a fix to a multi-hour production job.

**Important caveat on autonomy:** the user gave verbal authorization
*within that conversation* to execute steps 2-4 automatically, without
re-asking, contingent on each step succeeding and on Claude's own judgment
that it's ready. That authorization is scoped to the session it was given
in - per this project's standing preferences (section 6 below), a new
session should treat it as historical context explaining intent, not as
standing permission, and should confirm before executing multi-hour
cluster actions itself.

**Also important - the automation didn't survive.** A background watcher
was set up in the previous session to detect job `1760803` finishing and
auto-continue the chain. The user closed VS Code to go do other work in a
local devcontainer; the background task was torn down without a
completion record (`status: stopped`, no output). **Nobody is currently
watching this job.** Whoever picks this up needs to manually check
`squeue -j 1760803` and re-arm monitoring, or just check back periodically.

## 4. What to check once the benchmarking run itself actually happens

(Carried forward from the original handoff, unchanged - still applies once
step 4 of the chain above actually runs.)

- `outputs/reports/run_summary.json` - top-level pass/fail per task.
- `outputs/reports/<task_id>/summary.json` - per-model entries include a
  `datasets` field with `{attempted, succeeded, error}` per dataset.
- `outputs/reports/<task_id>/<model>/cognitive_summary.json` - for
  OECR/SRC, confirm the `llm_judge` block got merged in.
- Actual wall-clock time for the whole job, and a per-model boot-time
  breakdown if visible in logs - first real data point for calibrating
  `--time` and `startup_timeout_s` values, which have been guesses/
  estimates until now (see the `--time` bump in section 2 above, itself
  still an estimate).
- Note: `outputs/` already contains stale data from a June 27 smoke test
  (`cdkr`/`oecr` only, 2 models: `gemma3_27b_it`, `medgemma_27b_it`) - the
  real run will overwrite/mix with it, which is fine per the user (see
  section 6 - per-run output separation is a known future decision, not
  yet implemented).

## 5. Backlog - condensed, self-contained (carried forward from the last
handoff, don't assume memory access here)

**Architectural decisions, yours to call:**
- OECR's `reasoning_validity` judge rubric grades on medical soundness,
  not strict reference-matching - a deliberate, previously-made call, kept
  open because it was flagged as "worth revisiting."
- Reports-per-run vs. overwrite-in-place (`outputs/reports/` currently
  overwrites every invocation) - actively being decided, not just noted.
- Judge model selection is still architecturally open beyond this run:
  this session only decided to download `DeepSeek-V4-Flash` first to save
  time, not that it's the final answer over `gpt-oss-120b`/`gemma-3-4b-it`.

**Potential improvements, low priority, only after everything else:**
- Fewshot examples are concatenated into one flat chat turn instead of
  real multi-turn message history - a real interface change, not urgent.
- API-hosted models (Gemini specifically, for its free tier) - verified
  Gemini's OpenAI-compatible endpoint supports everything the existing
  `OpenAIBackend`/`JudgeClient` classes need; the actual blocker is that
  `orchestrator.py`/`llm_judge_run.py` unconditionally boot a local vLLM
  server for every model, with no path to just point at an external API.
- LLM-jury (3-judge majority vote instead of one judge) - architecturally
  cheap to add, but 3x judge-pass compute cost is the real blocker, plus
  an unresolved tie-break rule for genuine 3-way label disagreement.

**Known upcoming work, not yet scoped:** the user has flagged "a major
task regarding output analysis" they intend to work on before concluding
this backlog. No further detail exists yet - don't guess at its shape,
just be aware it's coming.

**Final step, deferred until everything above is done:** a dead-code
sweep, scoped to the Python side (`llm_bench/`,
`scripts/orchestration/*.py`) - the SLURM `.sh` scripts have now had two
dead-code/consistency passes (the original one, plus this session's
`python_vllm.sh` split and `HF_HOME` fixes).

## 6. Standing preferences (carried forward, still apply)

- Ask before editing shared/production config for local convenience; fix
  genuine, objectively-wrong bugs directly without asking.
- Don't silently drop or "helpfully" resolve backlog items - surface them.
- Verify claims empirically. Every fix in this project's history was
  validated by actually running something and checking real output, not
  just reasoning about what should happen.
- Never commit without being explicitly asked, even when a commit would
  obviously be useful (see section 0 - this session ended with real,
  reviewed, uncommitted changes and deliberately did not commit them).
- New this session: don't treat cross-session verbal authorization (e.g.
  "you can launch X automatically once Y succeeds") as carrying forward
  into a new session/context - it explains prior intent, but a fresh
  session should still confirm before taking equivalent multi-hour/
  resource-consuming cluster actions itself.
- New this session: background watchers/monitors do not survive the
  session/IDE closing. Don't promise continued autonomous monitoring
  across a session boundary - the next session (or a resumed one) needs to
  manually re-check state.

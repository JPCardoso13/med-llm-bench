# Handoff: moving to the rtx4060/haslab SLURM cluster

Written at the end of a long devcontainer session, for whichever Claude Code
instance picks this up on the cluster. Everything below is committed and
pushed - `git log`/`git status` are authoritative for what's actually in the
repo; this file is for the context that isn't obvious from reading code
alone (why decisions were made, what's still open, what to watch for).

## What this tool is

`llm_bench` is a config-driven, model- and domain-agnostic LLM benchmarking
framework, built as a Python library for a Biomedical Engineering (Medical
Informatics) master's thesis. It evaluates LLMs from both a **systems**
angle (TTFT, throughput, token usage, GPU telemetry, generation perplexity)
and a **cognitive** angle (accuracy/precision/recall/F1 for MCQ, token-F1/
ROUGE/exact-match for free text, plus an LLM-as-judge pass with categorical
rubrics - never numeric scores).

The core design principle - the thing basically all of the last session's
work was in service of - is that **the framework itself carries zero
hardcoded task-specific values**. A user defines datasets, prompts, and
which metrics to compute entirely in YAML; the Python code stays generic
across models, tasks, and datasets. The one real boundary: **new metrics
require code** (they need to be added to a calculator), but new
datasets/models/prompts should never require touching Python. If you find
yourself editing `llm_bench/` to support a new dataset or model, something's
wrong - that should be a config change only.

Three benchmark tasks, each pinning the framework to the thesis's
biomedical scope:
- **CDKR** (Closed Domain Knowledge Retrieval) - MCQ, medical exam
  questions (MedQA, MedXpertQA)
- **OECR** (Open-Ended Clinical Reasoning) - free-text generative, clinical
  case reasoning (MedCalc-Bench, MedCaseReasoning)
- **SRC** (Summarization and Reading Comprehension) - free-text generative,
  biomedical document summarization (PubMedSum, MultiClinSum)

A vision task was floated as a stretch goal but is explicitly not
committed scope - don't build it unless the user asks.

This serves two purposes for the thesis: (1) a real end-to-end validation
of the framework itself, and (2) actual conclusions about LLM behavior in
the biomedical domain, which the thesis needs regardless of the tool.

## Where things stand

The last long session did a full pass on config/code correctness across the
whole repo - `configs/metrics/`, `configs/datasets/`, `configs/tasks/`,
`configs/prompts/`, `configs/models/`, `configs/runs/`, and most of
`scripts/` (orchestration, vllm launch shim, asset caching, sanity checks).
Real bugs were found and fixed along the way, not just theoretical cleanup -
a couple of the more consequential ones: a calculator silently ignoring an
entire config block because the code still looked for a renamed key: fixed;
`download_hf_assets.py` resolving its own project root one directory level
too shallow, meaning it would silently discover zero models/datasets and
report "download complete" having downloaded nothing: fixed; a dead
monkey-patch (`scripts/patches/`) that had never actually loaded due to a
separate path bug: removed entirely rather than fixed, since it wasn't
needed anymore anyway.

A 3-model local smoke test (Qwen2.5-3B, Granite-3.1-2B, Gemma-3-1B, one
model per vendor family) just ran clean end-to-end in the devcontainer -
all 3 tasks, 50 samples/dataset, including a real judge pass and real GPU
telemetry (via the `nvidia_smi` collector, which runs in-process with zero
external setup - see the telemetry note below for how this differs on a
SLURM cluster). This proved the whole pipeline works, but at devcontainer
scale on small models - it hasn't been proven on a real multi-model SLURM
run since the config/code fixes above landed.

Reporting/analysis (`scripts/analysis/generate_report.py` +
`llm_bench/reporting/`) is built and has been iterated on with real
feedback multiple times - it turns `outputs/reports/*.json` into charts and
tables under `outputs/analysis/`. **As of the last session, it's now wired
into `run_pipeline.sh` as a final automatic step** (it wasn't before - the
user explicitly wants one SLURM job to produce everything: benchmark
results, judge scores, and human-readable reports, with no manual
follow-up step). This addition follows the exact pattern of the existing
judge-pass step but has not yet been proven on a real cluster run.

## Why the move to this cluster specifically

Deucalion (the bigger A100 cluster) has a storage quota problem that isn't
resolvable soon, so the plan is to run on the smaller rtx4060/haslab
cluster instead. This is a real hardware downgrade, not a lateral move -
the current `configs/models/*.yaml` production roster (the 8 models in
`configs/runs/full_production.yaml`) was sized for Deucalion's 4x80GB A100
allocation and almost certainly does not fit here. The user believes ~4
GPUs at 16GB each may be available on this cluster, but this needs
confirming on the cluster itself before any model decisions are finalized.

A rough sanity check on the existing catalog against a 4x16GB (64GB total)
budget, for calibration - not a final answer, VRAM math should be verified
properly once real hardware specs are confirmed:
- The two 70B models (`llama3_1_70b`, `llama3_openbiollm_70b`, currently
  `tensor_parallel_size: 4`) need ~140GB unquantized even split across 4
  GPUs - won't fit.
- The unquantized 27B/35B models (`gemma3_27b_it`, `medgemma_27b_it`,
  `qwen3_6_27b`, `qwen3_6_35b_a3b`) are all sized for a single 80GB-class
  GPU or two - likely too large per-GPU here too.
- The two already-quantized configs (`qwen3_6_27b_awq_int4`,
  `qwen3_6_35b_a3b_awq_4bit`) are the closest existing candidates to
  actually fitting, though their current `tensor_parallel_size` (1 and 1)
  may still need adjusting for comfortable KV-cache headroom on 16GB cards.

**The user is choosing the actual model list themselves and will bring it
to this session** - don't guess at model choices. Once you have that list,
create a **new** `configs/runs/*.yaml` (don't overwrite
`full_production.yaml`, which represents the Deucalion-scale run) and add
matching `configs/models/*.yaml` entries sized correctly for whatever this
cluster's real GPU spec turns out to be. Same for the judge model - neither
current judge candidate (`deepseek_v4_flash_judge.yaml`, ~156GB;
`gpt_oss_120b_judge.yaml`, ~117B) was sized for this hardware; a
right-sized judge needs picking once the benchmarked-model list is settled.

## Task list, roughly in order

1. **Confirm real hardware** on the rtx4060/haslab cluster (GPU count,
   VRAM/GPU, node count available) - don't trust the "~4x16GB" guess above
   without checking.
2. **Work with the user to pick the benchmarked-model list and a judge
   model** sized for that hardware, and build the corresponding
   `configs/models/*.yaml` + a new `configs/runs/*.yaml`.
3. **Review `scripts/orchestration/slurm_orchestrator.sh`'s SLURM
   allocation** against the chosen models - it currently requests
   `--nodes=2` with only `--time=02:00:00` and no explicit `--gpus`, which
   was never verified against a real workload (unlike
   `deucalion_orchestrator.sh`, which explicitly matches its allocation to
   the production run). This needs a real decision once you know what's
   actually running, not a guess.
4. **Run a small validation pass first** - 50-sample limit (matching the
   scale already proven in the devcontainer), same models, to confirm
   telemetry and the pipeline generally work correctly on real cluster
   hardware before committing to the full run. Check `outputs/analysis/`
   afterward, don't just check for a zero exit code - look at the actual
   charts/tables and sanity-check the numbers.
5. **Decide the final sample size with the user** before the definitive
   run - every run so far, including the devcontainer smoke test, has used
   `eval_limit: 50`, which was explicitly smoke-test scale, not
   necessarily final-thesis scale.
6. **Run the definitive test.** One `sbatch` submission via
   `slurm_orchestrator.sh` now produces the full pipeline: benchmark ->
   judge -> analysis report, per the `run_pipeline.sh` change made last
   session. Confirm this actually happens end-to-end on a real cluster
   before treating it as reliable - it's real code, following the same
   pattern as the rest of the script, but has not been tested against a
   real SLURM environment.

## Things to know before you start

- **Ephemeral HF caching is already configured for this cluster**
  (`slurm_orchestrator.sh` sets `HF_CACHE_MODE=ephemeral`,
  `HF_EVICT_BETWEEN_MODELS=1`) - weights download fresh into a job-scoped
  temp dir and get evicted between models rather than accumulating in
  persistent storage. This was a deliberate choice given this cluster's
  project-folder quota (the user mentioned ~50GB) - don't "fix" this into
  persistent caching without checking whether the quota problem is still
  real.
- **Telemetry requires zero manual setup on SLURM** - `run_pipeline.sh`
  already launches `scripts/telemetry/nvidia_smi_agent.py` on every
  allocated node and auto-generates `configs/runtime/telemetry.auto.yaml`
  with `collector: remote_http` pointing at them. This is different from
  the devcontainer path (which uses the in-process `nvidia_smi` collector
  instead) - the two collectors produce differently-named output fields,
  and `configs/metrics/systems.yaml`'s `telemetry.fields` list already
  covers both shapes, so this shouldn't need touching either way.
- **Run with `bash`, never `sbatch` directly** on `slurm_orchestrator.sh` -
  it submits the real job itself. Submitting `run_pipeline.sh` directly
  will fail loudly (`WORKDIR` is required and unset outside the wrapper).
- **Don't hand-author `configs/runtime/telemetry.auto.yaml`** - it's
  generated fresh per run by the shell script.
- Model identity throughout the framework comes from **config file name**,
  not any field inside the file - `name:` fields were removed from every
  model config last session because they were confirmed dead (never read).
  Don't reintroduce them as if they matter.
- If you find yourself about to hardcode a dataset/task/field name into
  Python or into `scripts/analysis/generate_report.py`'s logic (as opposed
  to its small curation constants at the top, which are meant to be
  edited), stop - that's very likely the wrong layer. The project's
  explicit standing preference is that this kind of curation belongs at
  the data source (a dataset's `mapping.grouping` vs `mapping.metadata`,
  for instance) or in a clearly-labeled constant, not buried in generic
  code.

## Backlog

- **Reports-per-run vs. overwrite-in-place**: `outputs/reports/` currently
  gets overwritten on every invocation, no run history preserved. Open
  discussion, not yet decided.
- **BERTScore**: discussed as a possible additional generative-quality
  metric, deliberately deferred until after everything above. Worth
  raising with the user if there's time before the definitive run, not
  before.

Everything else that was previously tracked as backlog (older architectural
debates, potential improvements like multi-turn fewshot / API-hosted models
/ LLM-jury, and the template-file rebuild) was explicitly closed out by the
user at the end of the last session - don't resurrect these without them
raising it first.

## Standing preferences

- Ask before editing shared/production config for local convenience; fix
  genuine, objectively-wrong bugs directly without asking, but say clearly
  what was found and what was changed.
- Don't silently drop or "helpfully" resolve backlog items - surface them.
- Verify claims empirically. Nearly everything fixed in this project's
  history was validated by actually running something and checking real
  output, not just reasoning about what should happen - keep that bar.
- Don't treat cross-session verbal authorization as carrying forward into a
  new session - confirm before taking multi-hour/resource-consuming
  cluster actions.
- Background watchers/monitors do not survive a session/IDE closing - don't
  promise continued autonomous monitoring across a session boundary.
- Never commit without being explicitly asked.

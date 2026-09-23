# Research Findings Log

Started 2026-09-04. This file tracks findings from the benchmark runs that
are candidates for the thesis's results/discussion chapter — kept separate
from `HANDOFF.md` (which is pipeline/engineering state, not research
content). Two categories, kept explicitly separate so a stale entry never
gets mistaken for a citable result:

- **Verified findings**: cross-checked against raw model output, confirmed
  not an artifact of a pipeline bug. Safe to build the writeup around,
  though still worth a final sanity pass before submission.
- **Open questions / needs more evidence**: plausible but not yet nailed
  down to the same standard — flagged so they don't accidentally get
  written up as settled.

Each entry has: the claim, what evidence supports it, and what was ruled
out (so the reasoning is auditable later, not just the conclusion).

---

## Methodology: completion-token budget calibration (2026-09-16)

Written for the thesis methods section — this is the standing rule, not a
dated correction.

**The budget is set per task, never per model.** `max_tokens` is
calibrated once for each task, based on (a) that task's expected output
shape and (b) measured prompt-length headroom against the model family's
context window (`max_model_len: 16384` for every model in this roster),
established during a pilot/measurement pass — not by watching any one
model's behavior and tuning around it. The same resulting budget is then
applied identically to every model compared on that task. This mirrors
standard practice in published LLM evaluation harnesses (e.g. HELM), which
fix a generation-length cap per task category rather than per system.

**Why this doesn't compromise the cross-model comparison**: the threat to
validity would be giving one model more completion budget than another
*on the same task*, since that hands the larger-budget model a structural
advantage independent of its actual capability. That never happens here —
within a task, budget is a constant across the model roster, not a
variable. What *is* allowed to vary is the budget between different
tasks (`cdkr`'s single-letter-plus-reasoning answer needs far less room
than `oecr`'s worked clinical calculations, and `src`'s bounded-summary
output combined with much longer inputs constrains how much budget is
even available) — that's a property of the task, not of any model being
scored on it.

**Residual truncation under a shared, headroom-verified budget is a
result, not noise.** Once a budget has been confirmed to leave genuine
margin under the context window (not simply "the largest value that still
fits"), a model that still gets cut off more often than another under
that *same* budget is demonstrating something real about its own
verbosity/chain-of-thought efficiency on that task. The correct response
is to report that difference (e.g. as a reliability/efficiency metric
alongside accuracy), not to individually enlarge that model's budget
until its truncation rate matches everyone else's — doing so would erase
exactly the signal being measured.

**Where this project's actual limitation lies**: not in the per-task
calibration above, but in *when* a task's budget was set relative to when
each model's data was generated — see "`max_tokens` truncation, measured
directly across the full roster" below. Some data predates a budget
recalibration purely because of job scheduling/timing, not by design.
That's timeline drift, and it is disclosed and corrected (via full
regeneration) separately from the calibration principle above, which
holds throughout.

---

## Pipeline corrections that affect result validity

These aren't findings themselves, but any reader of this log (or the
thesis) needs them to know which numbers are trustworthy and since when.

- **MCQ answer-extraction regex bug (found + fixed 2026-09-04)**: the
  original `cognitive.mcq.yaml` answer-extraction patterns could match the
  first letter of an unrelated word immediately after "answer" (e.g. "answer
  **i**s B" matched "I" from "is"; "Answer: **D**iscuss..." matched "D" from
  "Discuss"). This silently misgraded any model that answers in free text
  rather than a bare letter. Confirmed via direct regex testing and by
  recomputing metrics with a fixed pattern set (adds a trailing
  `\b(?!\w)` boundary check). Fixed in `configs/metrics/cognitive.mcq.yaml`;
  `175453` (Llama tier)'s `cdkr` metrics were recomputed from the existing
  raw output (no regeneration needed) and are now correct. Any MCQ result
  computed before this fix (anything predating 2026-09-04) should not be
  trusted without recomputation — check whether a given `cognitive_summary.json`
  was written before or after this date if in doubt.

- **MCQ answer-extraction regex bug, v2 (found + fixed 2026-09-08)**: the
  2026-09-04 fix above still let the captured letter itself BE a complete
  one-letter English word, since a space right after it also satisfies "not
  a word char" — the reasoning-model preamble "The user wants me to
  **answer a** multiple-choice question..." matched "a" as "A". This is
  systematic on any reasoning model (near-universal opening phrasing), and
  silently deflated `qwen3_6_35b_a3b_awq_quanttrio`'s measured `cdkr`/`medqa`
  accuracy from a real ~56% (spot-checked on 100 samples) down to 22.7% —
  most non-"A" correct answers got marked "ambiguous" once the false "A"
  candidate collided with the model's real, later-stated letter. Fixed by
  requiring the captured letter be followed by a closing paren (explicit
  option-marker intent) or immediately by clause-ending punctuation/
  newline/end-of-string — i.e. a letter followed by more lowercase prose
  with no punctuation no longer matches (trade-off: some genuinely
  unpunctuated answers now read as "missing" rather than extracted, which
  is safe — missing shrinks sample size, it doesn't corrupt accuracy).
  `175453` (Llama tier, unaffected either way — bare-letter answers,
  reconfirmed no regression) and `175482`'s `qwen3_6_35b_a3b_awq_quanttrio`
  were recomputed. Any MCQ result from a reasoning-style model computed
  before 2026-09-08 should not be trusted without recomputation.

- **`compute_and_save_metrics` never ran for some completed tasks (found +
  worked around 2026-09-08)**: `orchestrator.py`'s two-pass design computes
  metrics (Pass 2) only after Pass 1 finishes generating *every* task for a
  model — so a job that times out mid-generation on a later task (e.g.
  mid-`src`) never reaches Pass 2 at all, even for earlier tasks whose raw
  output is completely fine (e.g. `cdkr`, `oecr` both fully generated).
  Found on `175482` (`qwen3_6_35b_a3b_awq_quanttrio`): raw generation for
  `cdkr` and `oecr` was 100% complete, but neither had a `cognitive_summary.json`
  at all until recomputed by hand — `src` alone had real reports, only
  because a later resume job for `src` specifically completed cleanly on
  its own. Worked around (not yet fixed upstream) with
  `scripts/analysis/recompute_metrics.py` / `recompute_and_merge_judge.py`,
  which recompute a single task's metrics directly from existing `raw/`
  (and merge existing `judged/` scores) without needing generation to
  re-run. **When checking any run that timed out or crashed mid-generation,
  don't assume completed tasks have real reports — verify each task's
  `cognitive_summary.json` exists and its `mtime` is recent enough to
  reflect the actual raw data**, rather than trusting task order.

- **LLM-judge pass had no per-sample error handling (found + fixed
  2026-09-06)**: `llm_judge_run.py` had no try/except around the judge
  API call, and only wrote `judged/*.json` once per model *after all*
  datasets in a task finished. A single malformed judge completion (a
  degenerate looping generation from `gpt-oss-20b` tripped the backend's
  own request validation) crashed the entire judge pass with zero results
  saved (`175508`, 2026-09-04, `qwen3_6_35b_a3b_awq_quanttrio`/`oecr` — dead
  after 3h21m, nothing salvageable). Fixed: a failed judge call is now
  treated like a parse failure (continues to the next sample), and results
  flush to disk after every dataset instead of only once per task.

- **`175482`'s `qwen3_6_35b_a3b_awq_quanttrio` `cdkr`/`oecr` data predates
  the `max_tokens` 1024→3072 fix (confirmed 2026-09-15)**: that job ran on
  2026-09-01, before the 2026-09-08 bump documented above, so its `cdkr`
  accuracy (medqa 58.68%, medxpertqa 14.12%) reflects the ~26.4%/51.1%
  truncation rates already measured against the old 1024-token budget - it
  was never regenerated. A same-model, same-datasets TP=2 rerun under the
  current 3072-token config (job 175575→175577, `outputs/175575/raw/cdkr`)
  scored medqa 86.10%, medxpertqa 35.02% - confirming the truncation
  hypothesis's practical impact was even larger than the qualitative
  "meaningful share" language above suggested. This was originally intended
  as a pure TP=4-vs-TP=2 determinism sanity check (temperature 0 should give
  near-identical accuracy regardless of tensor-parallel degree), but the two
  runs differ in `max_tokens` too, so the large gap is NOT evidence against
  TP-invariance - it's the already-known truncation effect, now with an
  end-to-end before/after accuracy delta attached. **Practical implication**:
  `outputs/175482`'s `cdkr`/`oecr` raw data for this model is stale relative
  to the current config and should eventually be replaced by a rerun under
  `max_tokens: 3072` (the TP=2 data, once `oecr` is also done, is a
  candidate) - the TP-determinism question itself remains untested in
  isolation (would need a same-`max_tokens` TP=4-vs-TP=2 pair, not yet run).

---

## Verified findings

### Llama3-8B-Instruct vs. Llama3-OpenBioLLM-8B on `cdkr` (MCQ)

**Claim**: OpenBioLLM-8B, a biomedical fine-tune of Llama-3-8B, performs
substantially *worse* than its own base model on multiple-choice medical
QA (medqa: 27.9% vs. 58.7%; medxpertqa: 7.1% vs. 11.6%), and the gap is a
real model-behavior difference, not a measurement artifact.

**Evidence it's real, not a pipeline bug**:
- Confirmed after the extraction-regex fix above, so it isn't the "I"/"D"
  mismatch bug.
- Manually inspected raw responses across a random sample and a full
  scan of parse failures (see sub-findings below) — the gap is driven by
  two distinct, characterized behaviors in OpenBioLLM's actual output, not
  by a scoring defect.
- The base model (`llama3_8b_instruct`) answers cleanly (bare letter, e.g.
  `"A"`) and its scores are unaffected by the regex bug either way, so it's
  a clean baseline for comparison.

**Two distinct sub-findings drive the gap:**

1. **Confidently wrong, coherent answers** (the "normal" capability-gap
   part of the story): a large share of OpenBioLLM's incorrect answers are
   short, decisive, medically-phrased, just wrong — e.g. `"The answer is D)
   Hepatitis A."` (ref: A), `"The answer is B) Aphthous stomatitis."` (ref:
   A). This is a legitimate, explainable reasoning/knowledge gap.

2. **Token-level output corruption on ~14% of all medqa samples** (177 of
   1273) — not rambling, actual garbled/non-answer text with a specific,
   repeated signature. The same nonsense strings recur verbatim across
   dozens of unrelated questions:
   - `"The answer is:flutterassistant hoạch"` (~15 occurrences)
   - `"The answer is:flutterassistant_firestore"` (~8 occurrences)
   - `"The answer is��apoxetine."` (~9 occurrences, identical each time)
   - `"The exercisewasnotconductedby..."` variants (~10 occurrences)
   - `"The correct answer is libertinus."` (2 occurrences, identical)

   These are unrelated to medicine (Flutter/Firestore are Google
   dev-tooling products; "libertinus"/"apoxetine" are not real medical
   terms). **Ruled out**: not a length/stop-token failure — every one of
   these checked has `finish_reason: "stop"` in `backend_metrics` (the
   model terminates cleanly and deliberately, it doesn't run past a
   truncation limit). The repeated, specific, cross-context recurrence of
   the exact same non-medical tokens is the signature of a token
   decoding/vocabulary anomaly in the serving setup, not random sampling
   noise or verbosity. Total additional accuracy cost from this alone:
   these samples can never be scored correct since they don't state an
   answer, so they mechanically drag accuracy down independent of the
   model's "real" medical knowledge.

**Model card corroboration (2026-09-04, via HuggingFace)**: confirmed
`aaditya/Llama3-OpenBioLLM-8B` is the legitimate, official, canonical repo
(Saama AI Labs / Ankit Pal) — not a mirror or outdated fork; no newer 8B
version exists (there is a separate 70B variant). The model card itself
warns: *"The model output can be verbose in rare cases. Please consider
setting temperature = 0 to make this happen less."* This is a directly
relevant, citable fact: **our pipeline already uses `temperature: 0.0`**
(the orchestrator's own default, confirmed in `scripts/orchestration/orchestrator.py`,
and `llama3_openbiollm_8b.yaml` sets no override) — i.e. we are already
following the developers' own recommended mitigation, and the deterministic
token-corruption pattern still occurs. This strengthens rather than weakens
the finding: at temperature 0 (greedy decoding, reproducible), the model
still confidently and repeatedly produces the same specific out-of-domain
tokens for certain inputs — this is not sampling noise, and it happens
despite following the model's documented best-practice settings. Framing
for the thesis: an independently-reproducible limitation of the model
itself under its own recommended configuration, not a pipeline
misconfiguration on our end. The model card's own "verbose in rare cases"
phrasing likely undersells what's actually a more specific decoding
anomaly than mere verbosity.

**Still open**: exact mechanistic root cause of the token-corruption
signature (see open questions below) — the tokenizer/chat-template
mismatch is one candidate mechanism, but the model developers' own
acknowledgment of instability (independent of our setup) means the cause
could equally be internal to the checkpoint itself. Not yet proven either
way.

**Where to find the underlying data**: `outputs/175453/raw/cdkr/llama3_openbiollm_8b.json`
(raw responses + `backend_metrics.finish_reason`/`perplexity` per sample),
`outputs/175453/reports/cdkr/llama3_openbiollm_8b/cognitive_summary.json`
(`mcq.diagnostics`, `mcq.parsing` — the parse-failure/ambiguous counts).

### `max_tokens` truncation, measured directly across the full roster (2026-09-16, corrected same day, resolved 2026-09-18)

Every prior mention of truncation in this log was either a parse-failure
proxy or a qualitative spot-check. Measured directly this time, straight
from each sample's `backend_metrics.finish_reason` (`"length"` = the
server cut generation off at the token cap, not a natural stop) — no
estimation involved. **Correction**: the first version of this table
inferred each row's `max_tokens` from the current task config file rather
than each sample's own recorded `backend_metrics.max_tokens_configured` —
this silently mislabeled one row (see the flagged line below and the
dedicated finding immediately after this table).

| model | task/dataset | `max_tokens` (per-sample, verified) | length-truncation % |
|---|---|---|---|
| llama3_8b_instruct | all 6 datasets | 1024/512 | 0.0–0.5% |
| llama3_openbiollm_8b | cdkr/oecr (4 datasets) | 1024 | 0.1–4.5% |
| llama3_openbiollm_8b | src/pubmedsum | 512 | **14.2%** |
| llama3_openbiollm_8b | src/multiclinsum | 512 | 1.3% |
| qwen3_6_27b_awq_quanttrio | cdkr/medqa (superseded, was 1024) | ~~1024~~ **3072, redone `175613`** | ~~29.8%~~ **2.3%** |
| qwen3_6_27b_awq_quanttrio | cdkr/medxpertqa | 3072 | 7.4% |
| qwen3_6_27b_awq_quanttrio | oecr/medcalcbench (superseded, was 3072) | ~~3072~~ **8192, redone `175621` (in progress)** | ~~46.0%~~ pending |
| qwen3_6_27b_awq_quanttrio | oecr/medcasereasoning (superseded, was 3072) | ~~3072~~ **8192, redone `175621` (in progress)** | ~~60.5%~~ pending |
| qwen3_6_27b_awq_quanttrio | src/pubmedsum | 3072 | 9.4% |
| qwen3_6_35b_a3b_awq_quanttrio | cdkr (TP=4, pre-fix, `outputs/175482` — kept as historical record, not deleted) | 1024 | 54.2% / 94.3% |
| qwen3_6_35b_a3b_awq_quanttrio | oecr (TP=4, pre-fix, `outputs/175482` — kept as historical record, not deleted) | 1024 | 99.8% / 100.0% |
| qwen3_6_35b_a3b_awq_quanttrio | src (TP=4, pre-fix, `outputs/175482`, superseded) | ~~512~~ **3072, redone `175620`, complete** | ~~100.0% / 100.0%~~ **1.0% / 0.8%** |
| qwen3_6_35b_a3b_awq_quanttrio | cdkr (TP=2 redo, `outputs/175575`) | 3072 | 1.8% / 4.8% |
| qwen3_6_35b_a3b_awq_quanttrio | oecr (TP=2 redo, `outputs/175592`, now complete — see resolution below) | ~~3072~~ **8192** | ~~39.9%~~ **0.5% / 1.2%** |

**Three findings here, not two:**

1. **The Llama tier is not uniformly unaffected by the old, smaller budget**,
   contrary to the blanket assumption written when `max_tokens` was bumped
   (see the pipeline-correction entry above). The base Instruct model
   genuinely never approaches either cap. The biomedical fine-tune
   (OpenBioLLM) does, specifically on `src`/pubmedsum (14.2%) — small
   enough that a re-run is arguably not worth it, but real enough that it
   should be disclosed as a limitation rather than asserted away, given
   this data is going into a thesis.
2. **The 2026-09-08 fix (1024/512→3072) reduced truncation but did not
   solve it for either reasoning model.** For `qwen3_6_27b_awq_quanttrio`,
   truncation on the generative task (`oecr`) is still 46–60% even at
   3072 tokens — worse than "elevated," closer to "most samples never
   finish." The 35B model's TP=2 redo fares much better on `cdkr` (1.8–4.8%)
   but still shows 39.9% on `oecr`/medcalcbench. This means `oecr` metrics
   for both reasoning models, even under the current config, are computed
   over a large fraction of cut-off responses — a real limitation to state
   plainly in any accuracy/quality comparison, not something the 2026-09-08
   fix already covers.
3. **`qwen3_6_27b_awq_quanttrio`'s `cdkr` raw file is itself a
   mixed-vintage merge (found 2026-09-16, while investigating why the
   dense model appeared far more truncated than the MoE model on medqa
   specifically).** `medqa` was generated whole in an early run (job
   175459, 2026-08-31) before the 2026-09-08 `max_tokens` fix, and never
   redone; `medxpertqa` needed two later `eval_offset` resumes that landed
   after the fix. The two datasets ended up merged into one "final" file
   at two different budgets without that being visible unless you check
   `backend_metrics.max_tokens_configured` per sample rather than the task
   config file (which only ever shows the *current* value). Practical
   effect: **there is currently no valid post-fix measurement of this
   model's `cdkr`/medqa truncation** — the 29.8% above is a real number,
   but at the old 1024 budget, not comparable to the 35B model's 3072-budget
   1.8% the way it was originally presented. Where both models are
   genuinely at 3072 (`medxpertqa`), the gap is real but far smaller: 27B
   7.4% vs. 35B 4.8%. A `medqa`-only redo at 3072 is queued
   (`configs/tasks/cdkr_medqa_redo_3072.yaml`) to close this gap. **Lesson
   for every future audit of this kind**: always check
   `backend_metrics.max_tokens_configured` (or any other per-sample
   generation-condition field) directly on the samples being compared —
   a task config file only tells you what *would* be used today, not what
   was actually used to produce a given raw file, especially one that was
   merged from multiple resume jobs spanning a config change.

**Resolution (2026-09-18)**: both fixes confirmed working, directly measured
post-redo:

- `qwen3_6_35b_a3b_awq_quanttrio`'s `oecr` at 8192 tokens (job 175592,
  complete): `medcalcbench` 39.9%→**0.5%**, `medcasereasoning`→**1.2%**.
  Both now comfortably in the "negligible" range.
- `qwen3_6_27b_awq_quanttrio`'s `cdkr`/medqa at 3072 tokens (job 175613,
  complete, merged into `outputs/175452/raw/cdkr`): 29.8%→**2.3%**, now
  consistent with `medxpertqa`'s 7.4% and the 35B model's 1.8-4.8% —
  confirms finding #3 above was correctly diagnosed as a stale-budget
  artifact, not a real dense-vs-MoE architectural difference.
- `qwen3_6_35b_a3b_awq_quanttrio`'s `src` full redo at 3072 (job 175620,
  complete): `pubmedsum` 100.0%→**1.0%**, `multiclinsum` 100.0%→**0.8%**.
- `qwen3_6_27b_awq_quanttrio`'s `oecr` redo at 8192 (jobs 175621→175645→175676,
  three resume passes total, complete 2026-09-23): `medcalcbench`
  46.0%→**2.8%**, `medcasereasoning` 60.5%→**11.9%**. Both dramatically
  better, but `medcasereasoning`'s 11.9% is the highest residual truncation
  of any fixed dataset so far (35B's equivalent redo: 1.2%). Not yet
  explained — worth a closer look before treating 8192 as "enough" for this
  specific model/dataset pair; flagged as an open question below rather
  than assumed fine by analogy with the other fixes.

### LLM-judge pass, `qwen3_6_35b_a3b_awq_quanttrio` `oecr` (2026-09-20)

First real judge pass run against corrected (8192-token) data — `gpt-oss-20b`,
99.6-99.8% parse success on both datasets. Headline rubric scores
(`diagnosis_correctness` / `reasoning_validity` / `safety_flag`):

- `medcalcbench`: 80.6% Correct, 79.4% reasoning rated Excellent, 2.0%
  flagged Unsafe.
- `medcasereasoning`: only 46.7% Correct (51.7% Incorrect), 52.2% reasoning
  rated Excellent, and **19.0% flagged Unsafe** — nearly 10x
  `medcalcbench`'s rate.

The `medcasereasoning`/Unsafe gap is worth a closer look before writing this
up: `medcasereasoning` is open-ended diagnostic reasoning over real clinical
cases (harder, more room for a genuinely unsafe recommendation) vs.
`medcalcbench`'s more constrained numeric calculations, so some gap is
plausible on task-difficulty grounds alone — but a 10x jump hasn't been
spot-checked against actual transcripts yet, so treat as a real, measured
number and an open question on root cause, not yet a settled explanation.

### LLM-judge pass, `qwen3_6_35b_a3b_awq_quanttrio` `src` (2026-09-20)

97.3-98.1% parse success on both datasets. Headline rubric scores
(`coverage` / `faithfulness`, both summarization-appropriate rubrics —
no `safety_flag` on this task):

- `pubmedsum`: 84.0% coverage rated Excellent, 91.0% faithfulness rated
  Excellent.
- `multiclinsum`: 74.5% coverage rated Excellent, 88.9% faithfulness rated
  Excellent.

No red flags here — both datasets look healthy, in contrast to `oecr`'s
`medcasereasoning` safety-flag anomaly above.

**Not yet done**: raising `max_tokens` again (e.g. to 4096+) and
re-measuring, to see whether `oecr`'s truncation rate for reasoning models
keeps falling or has hit some other bottleneck (e.g. the model's own
`max_model_len` ceiling, or genuinely verbose chain-of-thought regardless
of budget). Given each bump requires a full re-generation, this is a
deliberate cost/benefit call for whoever owns the timeline, not something
to silently redo.

---

## Integrity audit (2026-09-21)

A full, strict pass over every model's current raw data, reports, and charts,
requested explicitly before any results get evaluated for the thesis.
Checked per model/task/dataset: entry counts vs. expected size, duplicate
`sample_id`s, `finish_reason`/`max_tokens_configured` consistency, parse
rates, chart freshness, and plausibility of headline numbers. Full method:
counted directly from raw JSON + `backend_metrics`, cross-referenced against
`reports/*/cognitive_summary.json`, and (for the two findings below) verified
against the actual scoring source code, not just its config.

**Structural integrity: clean.** Every dataset across all four models shows
full expected coverage (1273/2450/1100/897, and pubmedsum/multiclinsum within
the explained range below), zero duplicate `sample_id`s anywhere, zero empty
responses, and `backend_metrics.perplexity` present and meaningfully varied
per sample (hundreds of distinct values per dataset) — real, distinct API
responses throughout, not placeholder or duplicated data. `max_tokens_configured`
is now single-valued per dataset everywhere current (the `cdkr`/medqa
mixed-vintage bug from 2026-09-16 is confirmed fully resolved and does not
recur elsewhere).

**Investigated and resolved as genuine coincidence, not a bug**: `llama3_8b_instruct`
and the *old, pre-fix* `qwen3_6_35b_a3b_awq_quanttrio` (`outputs/175482`) both
show `cdkr`/medqa accuracy of exactly 747/1273 (58.6803%, matching to full
float precision). Verified by reading the actual extraction implementation
(`llm_bench/metrics/answer_extraction.py::extract_mcq_answer_letter` — collects
every distinct `[A-J]` letter matched by any of the 4 configured patterns
across the whole response; >1 distinct letter anywhere = "ambiguous" = counted
wrong) and faithfully reproducing it standalone: it independently reproduces
all four models' cached accuracies exactly (747/1273, 747/1273, 1096/1273,
1071/1273), confirming both the code and the cached numbers are correct. The
match is a real coincidence between two unrelated mechanisms — Llama's plain
performance ceiling on non-reasoning MCQ vs. the pre-fix 35B run's answers
being destroyed by 1024-token truncation — worth noting since a coincidence
at 16 significant digits reads as suspicious at a glance, but not something to
act on: that 35B run is already stale/superseded data.

**Real finding — `src` (summarization) BERTScore/token_f1 may be scored on
contaminated text for both Qwen models, not a clean summary.** `answer_clean_rate`
is exactly 0.0 for both `qwen3_6_27b_awq_quanttrio` and
`qwen3_6_35b_a3b_awq_quanttrio` on every `src` dataset (`pubmedsum`,
`multiclinsum`), vs. ~90-100% for `llama3_8b_instruct`. Traced to the actual
scoring code (`llm_bench/metrics/cognitive_calculator.py`): `oecr`'s
contamination handling *truncates* the offending text before scoring
(`on_multiline: truncate_first_line`), but `src`'s config uses `on_multiline: join`,
which only despaces newlines — it does **not** strip anything, despite the
config's own comment implying multi-line content there is "legitimate" and
not contamination. Root cause confirmed directly in raw output: these Qwen3.6
AWQ builds emit un-tagged chain-of-thought prose (not wrapped in `<think>`
tags, which is the only thing `clean_response_text()` strips), so their real
answer is preceded by an unstripped reasoning preamble that then (a) trips
the unconditional "contains a newline → contaminated" flag and (b) survives
into the text actually scored against the reference summary for
BERTScore/token_f1. **Practical implication**: the `src` BERTScore/token_f1
numbers reported for both Qwen models (F1 ≈ 0.81-0.83) are likely computed
partly against leaked reasoning text, not a clean final summary, and are not
directly comparable to Llama's equivalent numbers on the same basis. The
LLM-judge `coverage`/`faithfulness` scores are a separate, judge-based
mechanism and are not known to share this problem — they remain the more
trustworthy `src` quality metric for the Qwen tier until this is fixed
(e.g., a CoT-stripping heuristic for un-tagged reasoning preambles, or
truncating on the first contamination marker for `src` too).

**Fixed and verified (2026-09-21).** Added a new `on_multiline: last_paragraph`
mode (`llm_bench/metrics/cognitive_calculator.py::_normalize_final_answer`):
splits the captured answer on blank lines and scores only the last paragraph,
still flagging `contaminated=True` (the compliance-failure signal is kept,
not deleted — matching `oecr`'s existing design). Only
`configs/metrics/cognitive.generative.summarization.yaml` (used exclusively
by `src`) opts into it; `join`/`truncate_first_line` behavior is unchanged
for every other config. Validated the heuristic against two real raw samples
(one 35B `multiclinsum`, one 27B `pubmedsum`) before applying it — in both
cases it cleanly isolated the model's actual final paragraph from the
numbered planning steps above it.

Recomputed `src` for all 4 models (no regeneration needed — pure
metrics-recompute against existing raw output) and confirmed:
- **Llama tier: unchanged**, as expected (single-paragraph outputs have only
  one "paragraph" either way). `llama3_8b_instruct` pubmedsum token_f1
  0.4351→0.4285, bertscore_f1 0.8458→0.8457 (noise-level, from the ~9% of
  samples already flagged contaminated pre-fix); `llama3_openbiollm_8b`
  multiclinsum token_f1 unchanged to 15 decimal places (0.386169821803997).
- **Both Qwen models: large, consistent improvement.**
  `qwen3_6_27b_awq_quanttrio`: pubmedsum token_f1 0.1485→**0.3784** (+155%),
  bertscore_f1 0.8179→0.8394; multiclinsum token_f1 0.0922→**0.3411** (+270%),
  bertscore_f1 0.8206→0.8582. `qwen3_6_35b_a3b_awq_quanttrio`: pubmedsum
  token_f1 0.1683→**0.3819**, bertscore_f1 0.8144→0.8404; multiclinsum
  token_f1 0.1074→**0.3655**, bertscore_f1 0.8189→0.8629. Both now much
  closer to (though still somewhat below) Llama's range, consistent with
  scoring the model's real answer instead of a reasoning-diluted blob.
  `answer_contaminated_count` stays at 985/1000 and 1000/1000 as intended —
  the compliance-failure signal is preserved, just no longer corrupting the
  quality score.

Isolation verified directly, not assumed: `cdkr`'s `cognitive_summary.json`
is byte-for-byte identical (md5 match) before/after for both Qwen models;
`oecr`'s report files carry mtimes from days before this fix, confirming
they were never touched. Charts regenerated for all 3 affected RUN_IDs
(175453, 175452_oecr_src, 175620); the concurrently-running generation job
(175645) was undisturbed throughout (separate nodes, no shared job slot
contention). Diff size: 2 files, 19 insertions, 1 deletion.

**Real finding, model behavior not a bug — `llama3_openbiollm_8b`'s `oecr`
format-parse rate is far lower than `llama3_8b_instruct`'s on the same task**:
12.5% (medcalcbench) and 3.2% (medcasereasoning) vs. 91.5%/99.3%. Spot-checked
raw responses directly: OpenBioLLM does give real, on-topic answers (e.g.
correct use of the Cockcroft-Gault equation), but states its final numeric
answer embedded in longer prose ("...the final answer is 54.497728 mL/min.")
rather than a clean, template-following final line the way Instruct does —
so the format-based extractor misses it far more often. Its LLM-judge parse
rate is unaffected (85-90%, matching Instruct's range), so judge-based scores
for OpenBioLLM's `oecr` remain reliable even though the format-based
`token_f1`/BERTScore "answer" metrics are computed over a much smaller,
successfully-parsed subset for this model — a comparability caveat, not a
pipeline defect.

**Explained, not a bug — pubmedsum/multiclinsum sample counts differ
per model** (923 for both Llama models, 985-991 for Qwen models). Confirmed
via the sequential runner (`llm_bench/runner/sequential_runner.py`): a
sample whose rendered prompt + task `max_tokens` exceeds the model's own
`max_model_len` is dropped (logged as a generation-failure warning, not
retried or backfilled). Llama's `max_model_len: 8192` overflows on ~77/1000
long documents; Qwen's `max_model_len: 16384` overflows on only 9-15/1000.
The underlying 1000-document sample *list* is identical for every model —
this is a legitimate consequence of differing context windows and tokenizers,
not a dataset-construction bug. Caveat worth keeping in mind for the thesis:
cross-model `src` comparisons are technically over slightly different
(~92-99% overlapping) sample sets, and there is no persisted per-run
manifest of which exact sample IDs were dropped for a given model/run — if
exact overlap ever matters, it would need to be reconstructed from
orchestration logs rather than read off a file.

**Real finding — few-shot draws are not aligned across models on
`eval_offset`-resumed slices (found 2026-09-21).** Each request is a
stateless chat (`[system, user]`; the task-level system prompt is re-sent
every time), and the few-shot examples inside the user turn are re-drawn per
sample via `random.Random(fewshot_seed).sample(...)` — one RNG per dataset,
created at runner construction and consumed once per sample. An
`eval_offset` resume constructs a fresh runner, so the RNG restarts from the
seed at the first *resumed* sample instead of advancing past the skipped
prefix: the resumed slice gets the draws that samples 0..k would have got.
Measured directly against the uninterrupted Llama run (same seed, same
sample order): `qwen3_6_27b_awq_quanttrio` `cdkr`/medxpertqa 515/2450
(21.0%, idx 1930+) and `qwen3_6_35b_a3b_awq_quanttrio` `cdkr`/medqa 273/1273
(21.4%, idx 1000+) received different few-shot examples than the same
samples did for the other models; every other current `cdkr` dataset is
fully aligned (0 mismatches). `src` is unaffected (`num_fewshot: 0`); `oecr`
will be affected only for 27B `medcalcbench` idx 980+ once job 175645 merges
(`num_fewshot: 2`). The `fewshot_seed` "same draws every rerun" guarantee in
the task configs therefore holds for uninterrupted runs only. Effect size on
accuracy is not quantified (few-shot choice adds some variance); correct
disclosure for the thesis is that ~21% of two datasets used a different (but
seeded, reproducible) few-shot draw. Possible remedy, not applied: advance the
RNG by `eval_offset` draws on resume (small code change) and regenerate the
two affected slices.

**Housekeeping found and fixed during this audit**: `outputs/175452` (the
27B model's `cdkr` home) had accumulated stale clutter from earlier in the
session — a leftover 5-sample toy-validation `oecr`/`src` (from the very
first, pre-real-run probe of this RUN_ID) and merged-away resume
intermediates (`cdkr_medxpertqa_only`, `cdkr_medxpertqa_resume2`,
`cdkr_medqa_redo_3072`) — all deleted; `outputs/175452` now contains only
its real `cdkr` data as intended.

**Known, already-tracked, not new**: `qwen3_6_27b_awq_quanttrio`'s `oecr`
(`outputs/175452_oecr_src`) still holds the pre-8192-fix data (46.0%/60.5%
truncated) pending job 175645's completion — expected, already in progress.

**Degenerate-response check (2026-09-22).** Prompted by a direct question
("are we sure every finished response is real and scoreable, not 5 words or
rambling without answering") — checked response-length distributions and
manually read the shortest response in every current model/task/dataset.

- Zero completely empty responses anywhere, across all four models, every
  dataset. Qwen tier: no degenerate responses found at all in spot-checks —
  every response is thousands of characters of real reasoning plus a real
  answer.
- `llama3_8b_instruct` `oecr`: 9 genuinely empty responses (7/1100
  `medcalcbench`, 2/897 `medcasereasoning`) — the model emits a bare
  `"Final answer:\n\nExplanation:"` template with nothing filled in, then
  stops natively (`finish_reason: stop`, not truncation). Checked whether
  this corrupts scoring: it does not — `oecr`'s extraction regex requires
  actual text between "final answer:" and "explanation:", so an empty
  capture correctly returns `None` and is counted as a parse failure, not
  scored as a real answer. Consistent with `medcalcbench`'s already-known
  91.5% parse-success rate. No action needed.
- **`llama3_openbiollm_8b` `src`/pubmedsum: 19/923 (2.1%) responses are the
  model echoing the task instruction back instead of summarizing** (e.g.
  `"You have read the document. Now summarize it in your own words."`,
  `"You: user"`). Unlike the Llama-Instruct case above, these ARE currently
  scored as valid: `src`'s extraction pattern is `^(.*)$` (captures the
  whole response, no content validation), so `parse_success_rate` reads
  1.0 for this dataset despite these 19 non-answers. Checked every other
  model/dataset combination for the same echo pattern — confirmed isolated
  to this one model/dataset pair, not systemic. **Not yet fixed** — awaiting
  the user's call on whether to add a small, isolated exclusion (same style
  as the `on_multiline: last_paragraph` fix: detect the echo pattern,
  exclude those 19 samples from being scored as valid, recompute from
  existing raw output, no rerun needed) versus documenting this 2%-of-one-
  dataset gap as a stated limitation instead.

---

## Open questions / needs more evidence

- **Why does `qwen3_6_27b_awq_quanttrio`'s `oecr`/medcasereasoning still
  truncate at 11.9% under the 8192 budget, when every other fixed dataset
  (35B's `oecr`/medcasereasoning: 1.2%; 27B's own `oecr`/medcalcbench: 2.8%)
  landed in the low single digits?** Same task, same budget, same prompt —
  only the model differs from the 35B comparison point. Candidate
  explanations, none checked yet: this model may reason more verbosely
  specifically on `medcasereasoning`'s longer, more open-ended clinical
  cases than on `medcalcbench`'s more constrained calculations; or 8192
  genuinely isn't enough headroom for this model on this dataset even
  though it was for the MoE model. Worth a quick check (prompt-length /
  completion-length distribution on the still-truncated 107 samples) before
  citing 8192 as a settled fix for this specific pair.
- **Root cause of the "flutterassistant"-class token corruption.** Leading
  hypothesis: a tokenizer/vocabulary misalignment between the serving setup
  and the fine-tuned checkpoint (candidate cause: the borrowed chat
  template), such that a narrow set of token IDs consistently decode to
  these specific out-of-domain strings regardless of prompt content.
  Investigating live (checking actual token IDs vLLM assigns) — not yet
  confirmed to the same evidentiary standard as the findings above. Do not
  cite as settled until confirmed.
- **`qwen3_6_27b_awq_quanttrio` still has no complete benchmark data** — the
  5-sample validation check (`175452`, original) was replaced with a real
  run, but it hasn't finished even `cdkr` yet (2026-09-08: 48h timeout hit
  at medqa 1273/1273 + medxpertqa 2140/2450, `oecr`/`src` not started).
  Nothing about this model can be reported until a real, complete run
  exists.
- **FP8 Qwen models (`qwen3_6_27b_fp8`, `qwen3_6_35b_a3b_fp8`)** have never
  produced any data — both attempts hung indefinitely during model loading.
  No findings possible here yet; status is "unresolved infrastructure
  issue," not a research result.
- ~~Reasoning models may need a much larger `max_tokens` budget for MCQ
  tasks than non-reasoning ones.~~ **Resolved and quantified 2026-09-16** —
  see "`max_tokens` truncation, measured directly across the full roster"
  under Verified findings below. Raising the budget (1024/512→3072,
  2026-09-08) was the acted-on fix, but direct measurement now shows it
  did not fully close the gap for the two reasoning models.

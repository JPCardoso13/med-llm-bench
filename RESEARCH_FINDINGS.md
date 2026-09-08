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

---

## Open questions / needs more evidence

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
- **Reasoning models may need a much larger `max_tokens` budget for MCQ
  tasks than non-reasoning ones.** On `qwen3_6_35b_a3b_awq_quanttrio`'s
  `cdkr` (generation `max_tokens: 1024`), a meaningful share of "missing"
  (unparseable) responses are genuine truncations — the model's chain-of-
  thought runs out of budget before ever stating a final letter, not an
  extraction failure (spot-checked: these responses end mid-reasoning,
  never reach "Final Answer" or similar). Not yet quantified precisely
  (need: fraction of parse-failures that are truncation vs. genuinely
  no-answer-given), and not yet acted on — raising `max_tokens` would
  require re-running generation, a deliberate cost/benefit call, not a
  silent fix. Worth deciding before writing up `cdkr` coverage numbers for
  any reasoning model.

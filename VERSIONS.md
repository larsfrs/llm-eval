# Version spec

A version pins everything a result row cannot see for itself: the content
behind existing question indices, the system prompts, and the answer
extraction rule. Sampling params need no version stamp; every row carries
its own, and all seven of them (temperature, top_p, top_k, min_p,
presence_penalty, repeat_penalty, max_tokens) are part of the resume
key, so changing any re-measures on its own. min_p/repeat_penalty joined
mid-v2; the existing rows were backfilled with the llama.cpp defaults
(0.0 / 1.0) they were implicitly measured with, so nothing re-measured.

Param columns record exactly what the model's params declare (values
overridden by the run where declared) — the payload carries the declared
knobs verbatim, nothing is injected anywhere. An empty param cell means
the knob was not sent and the server's own default applied (a provider
declaring only temperature/top_p/max_tokens leaves the other four cells
empty). Rows recorded under the old DEFAULTS-injected merge keep their
cells; the resume key re-measures only rows whose recorded params changed
(only Phi-4-mini, the lone partial-declarer).

Within a version, questions can be added but never changed or removed:
new indices appear as empty cells in results.csv and are filled
incrementally, one model at a time, as the models get served. Anything
that touches what is already measured is a bump; results from different
versions are not directly comparable.

The active version is the section marked "(current)" below and matches the
"version" key in eval.json. Bumping means: change the conditions in
eval.json / the prepare configs, bump the "version" key, and add a new
section on top. analysis.py only plots rows stamped with the active
version; a "(planned)" section is a sketch, never the active version.

## v3 (planned)

Sketch, not active — nothing here applies until eval.json's "version"
says v3. Problems the bump must solve:

- Budget: some questions starve inside max_tokens 4096 — thinking
  burns the budget before an answer is written. Pick the knob shape
  (per-model budgets, per-suite tiers, or a global bump) and raise it.
- Failure subtypes: cells record pass/fail only, so a wrong answer, a
  forgotten <answer> tag and a budget-cut look identical (the transport
  never reads finish_reason). Record why a cell failed — wrong /
  no-tag / cut — so budget and tag problems become measurable per
  model instead of console anecdotes.
- Tag compliance: some models forget the tags. Stronger system prompt,
  plus an audit of each model's think-tag variants: a model whose
  thinking special.py does not strip leaks drafted answers into the
  extraction channel. Grammar forcing stays out unless v3 deliberately
  stops measuring instruction following.
- Question growth: static is 5 questions, too thin to stand on — grow
  it and/or add rg cells so the bump carries one comparable set.
- rg re-roll: prepare/rg.py (SEED=42 today) moves to a new fixed seed
  — a different constant, not randomness at prepare time; the prepared
  file must stay frozen once generated. The bump re-measures everything
  under the new stamp, so the old v2 rg cells keep their rows.
(The model db item landed early, outside the bump: models.json + models.py
render the README table, and analysis.py reads the reference models' param
labels from the db.)

Deferred, considered and left out of v3: smp final form (repeats 1 vs
3, Phi-4-mini's non-sampled smp, the det +- 5*std aggregation), provider
hygiene (router pre-flight model check, remote-det determinism sanity),
housekeeping (gemma-4-26B-A4B orphan row, ask.py's debug print(payload)).

## v2 (current)

Changes from v1:
- max_tokens raised 2048 -> 4096 (det pin and model params), giving the
  harder rg tiers room for thinking
- rg slimmed from 30 to 8 questions: one per puzzle x difficulty --
  knights_knaves easy/medium/hard, cryptarithm easy/hard, zebra easy/
  medium/hard -- so 43 questions total instead of 65; later inside v2,
  jugs (easy/hard, 3 jugs, min 5/10 moves), graph_color (easy/hard, 3
  colors, 6/10 vertices), course_schedule (easy/hard, 5/10 courses) and
  self_reference (easy/hard, difficulty 2/7 of 10) added, rg now 16
  questions, 51 total; futoshiki held those cells first but was swapped
  out inside v2 (its library scorer caps correct-but-reformatted grids at
  0.9, a fail under the full-credit rule) and its cells were cleared

v2 in full:
- det: temperature 0.0, top_p 1.0, top_k 1, min_p 0.0, presence_penalty
  0.0, repeat_penalty 1.0, max_tokens 4096 (min_p/repeat_penalty joined
  mid-v2 with the server-default values, no behavior change)
- smp (when enabled): per-model provider params, repeats: 3, max_tokens 4096
- system prompts and extraction rule: unchanged from v1

## v1

### Questions -- 65 total
- static (5): hand-maintained, `data/static/questions1.json`
- linguistic_benchmark (30): fetched at prepare time from the
  [easy-problems-that-llms-get-wrong](https://raw.githubusercontent.com/autogenai/easy-problems-that-llms-get-wrong/main/linguistic_benchmark_multi_choice.json)
  repo, converted by `prepare/linguistic_benchmark.py`
- rg (30): reasoning-gym 0.1.25 (pinned in `uv.lock`), SEED=42, via
  `prepare/rg.py`, in difficulty tiers, easy -> hard:
  - knights_knaves: 4x easy (n_people=2, depth=2, width=2), 3x medium
    (n=3, depth=2, width=2), 3x hard (n=3, depth=3, width=3)
  - cryptarithm: 5x easy (2 addends), 5x hard (3 addends)
  - zebra_puzzles: 4x easy (3x3), 3x medium (4x4), 3x hard (5x5)

Regenerate the prepared files with:

    uv run prepare/linguistic_benchmark.py
    uv run prepare/rg.py

### Sampling params (as configured in eval.json)
- det (deterministic): temperature 0.0, top_p 1.0, top_k 1,
  presence_penalty 0.0, max_tokens 2048
- smp (sampled, when enabled): each model's provider-recommended params
  from its "params" block, repeats: 3, max_tokens 2048

### Also pinned
- system prompts (`eval/default.py`, `eval/rg.py`); a meaningful prompt
  change is a version bump
- answer extraction rule (`tools/extract.py`: last non-empty `<answer>`
  tag outside the thinking block)

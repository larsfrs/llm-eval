# Version spec

A version pins the two things that make results comparable across models:
the exact question set and the sampling parameters. Any change to either
one is a new version; results from different versions are not directly
comparable.

The active version is the topmost section below and matches the "version"
key in eval.json. Bumping means: change the conditions in eval.json / the
prepare configs, bump the "version" key, and add a new section on top.
analysis.py only plots rows stamped with the active version.

## v2 (current)

Changes from v1:
- max_tokens raised 2048 -> 4096 (det pin and model params), giving the
  harder rg tiers room for thinking
- rg slimmed from 30 to 8 questions: one per puzzle x difficulty --
  knights_knaves easy/medium/hard, cryptarithm easy/hard, zebra easy/
  medium/hard -- so 43 questions total instead of 65

v2 in full:
- det: temperature 0.0, top_p 1.0, top_k 1, presence_penalty 0.0,
  max_tokens 4096
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

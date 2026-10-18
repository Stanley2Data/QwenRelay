# Contributing

The first supported profile is dual RTX 3090 TP2. Keep single-card experiments and other model weights as separate, explicitly validated profiles.

For a source change, preserve upstream headers, explain the affected shapes/guards, and update before/after identities rather than weakening hash checks. A vLLM bump requires rebuilding and reviewing the entire inherited patch stack. Do not regenerate locks from an unexplained working tree.

Run `python3 scripts/qwenrelay.py check-pack`, `python3 -m unittest discover -s tests -v`, offline reconstruction against the pinned pristine source, and `bash -n` on changed Shell scripts. Numerical/GPU/performance claims require separate actual GPU evidence; CPU route tests do not establish kernel correctness. Keep regressions and unsupported shapes visible.

Submit source and small reproducible tests only. Keep model weights, venvs, build outputs, API keys, local paths, private logs and compilation caches out of commits. Benchmark comparisons must specify hardware, model SHA, effective profile, real input/output tokens, warmup/repeats and metric definition. Do not report `C / mean TPOT` as aggregate throughput.

There is no automated release/push pipeline in this initial source candidate. Public hosting and maintainer contact details will be established when the repository is published.

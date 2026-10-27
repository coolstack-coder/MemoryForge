# Contributing

Thanks for taking a look. MemoryForge is research code, so the most useful contributions are the ones that make results more trustworthy.

## Setup

```bash
make install
make test
```

`MF_OFFLINE=1` (set by `make test`) uses hashed embeddings, so the suite runs without downloads.

## Adding a benchmark task

1. Add a `Task` to one of the modules in `memoryforge/bench/`. Use a train module for experience tasks and `tasks_eval.py` for held-out ones.
2. Give it a precise spec, 6–10 hidden tests, a reference solution and 2–3 buggy variants. Tag each variant with a pattern from `memoryforge/failures_vocab.py`.
3. Every bug must fail **for the reason its pattern names**. For example, an `empty_input` bug should pass the non-empty tests.
4. Run `memoryforge validate` until it reports `0 problems`.

## Reporting results

Please keep simulated and real-model results in separate tables and label them. Include the seed count, the backend and model, the commit and the full `results.json`.

## Style

Standard library first, small functions, comments only where the *why* is not obvious. Keep commits small and focused.

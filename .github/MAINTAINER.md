# Maintenance

LangSlanger is currently maintained by [@evmcheb](https://github.com/evmcheb).

Pull requests should be small, use a Conventional Commit title, and include the
evidence required by `AGENTS.md`. The default checks are cheap. Add `run-ci` only
after the change is ready for the relevant GPU test matrix.

Merge only after required checks pass and the benchmark or correctness evidence
matches the risk of the change.

# Validation

Checked on 13 September 2026 using the local Python 3.13.13 environment.
Direct dependency versions are pinned in the requirements files.

| Check | Outcome |
| --- | --- |
| Synthetic regression tests | 14 passed |
| Ruff lint and formatting | Passed for Python modules, tools, tests and notebooks |
| Notebook structure | Six valid notebooks; outputs and execution counters cleared |
| Imports | All 18 root Python modules imported without starting an experiment |
| Command-line parsers | Seven experiment entry points accepted `--help` |
| Numerical comparison | 30 outputs matched the pre-refactoring baseline exactly on synthetic input |
| Real recording | First 30 ms of one local gesture produced finite tensors for all ten crop conditions |
| Dependency consistency | `pip check` reported no broken requirements |
| Export scan | No machine paths or common credential patterns found in exported text files |
| Git whitespace check | Passed |

The numerical comparison covered the ten crop conditions, retained-fraction
representations, GMM and appearance-tracker outputs, the blind-box diagnostic and
an overlap-ceiling calculation. The baseline was a snapshot of the working files before refactoring, rather than
an older Git revision.

These are regression and smoke checks. Full-dataset training, all notebooks,
complete tracking benchmarks, clean installation on another platform and the
remote GitHub Actions workflow were not run. The tests do not resolve the
scientific limitations listed in REVIEW_NOTES.md. The credential scan is a
pattern check, not a guarantee that every possible sensitive value is absent.

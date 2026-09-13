# FoveaNet

Research code for unsupervised foveation of event-camera recordings. A Gaussian
mixture identifies active regions in short time windows; accumulated component
weights select a rectangular crop. Fixed-budget classifiers compare the crop with
central, random, full-sensor and multiresolution inputs. Separate experiments
measure tracking against annotated targets.

## Start here

The verified local environment uses Python 3.13. Commands below run from the repository root. The pinned
dependencies match the local environment used for the cleanup checks; other
platforms and accelerator builds have not been verified.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Tests use synthetic events and temporary files; no dataset downloads or training
runs are needed. Dataset-based experiments require the layout in
[docs/DATASETS.md](docs/DATASETS.md). Keep datasets outside version control.

## Run an experiment

With DVS128 Gesture installed locally:

```sh
# A small preprocessing check: two recordings from each split.
python build_crops.py --window 10000 --limit 2 --out data/crops_smoke.npz

# Full preprocessing and five-seed final-epoch comparison.
python build_crops.py --window 10000 --out data/crops_review.npz
python classify.py --crops data/crops_review.npz --epochs 40 --seeds 5 \
  --val-subjects 0 --conditions full,persistence,persdens,size,density,centre,random,recentred,persist2x,fovperiph

# Activity diagnostic and animation.
python blind_box.py --dataset gesture --limit 30
python make_fovea_video.py --index 0 --rule persistence+hyst
```

The small preprocessing archive is not suitable for training or estimating
accuracy. The classifier's default condition list covers seven legacy conditions;
request the additional conditions explicitly as above. With `--val-subjects 0`,
use the **final-epoch** column, not the `MAX-TEST(old)` diagnostic or comparisons
computed from it. See [protocol notes](docs/REPRODUCIBILITY.md) before interpreting
results.

## Code map

| Files | Purpose |
| --- | --- |
| `foveanet.py` | Clustering, persistent GMM, saliency and box helpers |
| `aedat.py`, `tonic_convert.py` | Event-file readers and dataset adapters |
| `build_crops*.py` | Gesture, UCF-50, SL-Animals, DVS-Lip and area-sweep representations |
| `classify.py` | Shared 3D CNN and split-based evaluation |
| `blind_box.py` | Random-box/foveal activity ratio |
| `tracker.py`, `snn_tracker.py` | Local GMM and appearance-based tracking experiments |
| `evaluate.py`, `bench_tracking.py`, `gt_motion.py` | Ground-truth metrics and benchmark analysis |
| `make_figures*.py`, `make_fovea_video.py` | Figures and animation |
| `foveanet_01_*.ipynb` through `foveanet_06_*.ipynb` | Exploratory experiments and recorded summaries |
| `tests/` | Synthetic-data regression checks |

Notebooks stay at the repository root so their imports and relative dataset paths
work without path edits. Notebook outputs and execution counters are cleared.
Notebook 05 and parts of notebook 06 contain recorded result dictionaries: running
those cells displays historical measurements, rather than rerunning training.

`make_figures.py` similarly draws recorded constants. Its provenance is documented
in [report/figures/CAPTIONS.md](report/figures/CAPTIONS.md). The other figure scripts
read recordings. Existing figures are examples and are not automatically updated
when the code changes.

## Experiment records

- `FOVEANET_LOG.txt`: chronological experiments, failed approaches and corrections.
- `FOVEANET_PROGRESS.txt`: historical notes referenced by experiment identifiers.
- `PREDICTIONS.txt`: predictions and subsequent measurements.

These records are retained verbatim. Early claims can be superseded by later
entries; consult the protocol and dataset for each result. Numerical claims are
not validated merely by rerunning the plotting scripts.

## Review and reuse

See [docs/REVIEW_NOTES.md](docs/REVIEW_NOTES.md) for implementation limitations and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for evaluation details. This
repository has no licence grant yet; the owner should choose a licence before
advertising it for reuse. Datasets and third-party papers retain their own terms
and are not redistributed here.

## Prepare a fresh GitHub upload

The research checkout has an existing history containing old papers and local
authoring files. Removing them from the current tree does not remove historical
copies. To publish only the reviewed tree without rewriting that history:

```sh
python tools/export_review.py ../FoveaNet-GitHub
```

This creates a new repository on `main`, with no commits or remote configured.
It includes code, tests, docs, selected figures and research records. It excludes
local datasets, environments, caches and the old `.git` directory. Inspect that
folder before committing and adding your chosen GitHub remote.

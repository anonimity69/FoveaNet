# Reproducing comparisons

## Coordinate and timing conventions

Event arrays use `x`, `y`, `t`, `p`. Timestamps and window lengths are microseconds.
Sensor tuples are `(width, height)`; NumPy images use `(height, width)`. Crop tensors
have shape `(8, 2, 32, 32)`, or 16,384 values. `classify.py` transposes them to the
channel-first layout expected by a 3D convolution.

The shared exploratory window is 50,000 microseconds. For 10 ms experiments,
explicitly pass `--window 10000` to the crop builder. Existing archives may come
from earlier preprocessing; do not infer their protocol from the filename alone.

## Splits and epochs

Crop archives contain `label`, `user`, optional `split`, and one array per
condition. `split=0` is training and `split=1` is test. Without `split`, the
classifier holds out the last six sorted user IDs. This is a different protocol
from the supplied dataset split.

With positive `--val-subjects`, the last requested training subject IDs form a
validation set. The validation accuracy selects the epoch. The final-epoch test
accuracy is printed separately. Check that this leaves nonempty training,
validation and test sets, and that the subject metadata is meaningful.

`--val-subjects 0` trains on all training samples. The implementation also prints
a historical maximum-test diagnostic and uses that diagnostic for its subsequent
difference tables. Those tables are not unbiased final-epoch comparisons. Use the
final-epoch column for this protocol and calculate differences from those values.
The behaviour is retained here to avoid silently changing historical experiments.

NumPy and PyTorch are seeded, and preprocessing uses fixed NumPy seeds. GPU/MPS
kernels and dependency/platform differences can still change results. Full
cross-platform reproducibility has not been established.

## Retained-fraction sweep

```sh
python build_crops_sweep.py --window 10000 --out data/crops_sweep_review.npz
```

`fovNN` and `cenNN` use the same nominal box dimensions. The archive stores
`frac_<condition>` values because clipping can reduce actual area. Plot achieved
fractions. To train these conditions, pass their names explicitly to `classify.py`;
its default list is for the standard crop builder.

## Tracking

`python bench_tracking.py` runs the configured VOT and TrackingDataset evaluation.
`python gt_motion.py` summarises annotation motion. Both require local benchmark
HDF5 files. The appearance tracker is used through its Python API and exploratory
experiments, rather than as the default benchmark implementation.

## Recorded outputs

The log files and prediction record preserve the research chronology, including
withdrawn interpretations. Figure and notebook constants are summaries of those
runs. Lint checks and synthetic tests do not reproduce the full
multi-seed training experiments. Keep command lines, environment versions,
dataset splits and raw metrics with any new results.

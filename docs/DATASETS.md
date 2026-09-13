# Dataset layout

Run scripts and notebooks from the repository root. The loaders expect local
files beneath `data/`, which is ignored by Git. Download and redistribution terms
belong to the dataset publishers. The links below are those used in the project;
their present availability has not been checked as part of this cleanup.

```text
data/
  DVSGesture/
    ibmGestureTrain/  # preprocessed .npy recordings expected by Tonic
    ibmGestureTest/
  DVSBENCH/
    INI_VOT_30fps_20160610.hdf5
    INI_TrackingDataset_30fps_20160610.hdf5
    INI_UCF50_30fps_20160424.hdf5
  POKERDVS/
    cards_1.aedat
    ...
  SLANIMALS/
    SL-Animals-DVS_gestures_definitions.csv
    user*.aedat
    user*.csv
  DVSLip/
    DVS-Lip/
      train/
      test/
```

## DVS128 Gesture

The project uses Tonic's processed train/test layout, not arbitrary raw AEDAT
files. `foveanet.load_gesture` checks for at least 100 `.npy` files in the selected
split. It then bypasses Tonic's file-presence check for the local data. This uses
a private Tonic method and is one reason the dependency is pinned.

The original notebook used Figshare download identifiers 38022171 (training) and
38020584 (test); check the publisher's dataset listing before downloading. If
Tonic's automated download fails, download the processed archives manually and
extract the split directories into the layout above. Do not commit archives.

## DVS benchmark (Hu et al.)

VOT and TrackingDataset HDF5 recordings include `x_pos`, `y_pos`, `timestamps`,
`pol` and `bounding_box` datasets. Ground-truth rows contain a timestamp followed
by four corner coordinates. VOT sequences are top-level groups; TrackingDataset
sequences are nested. `DVSBenchmark` walks both layouts.

UCF-50 uses the same event fields but is loaded by `build_crops_ucf.py` for
classification. The benchmark sensor is 240 x 180.

## POKER-DVS

The project uses original `cards_*.aedat` recordings from the CAVIAR POKER-DVS
collection, rather than already-cropped samples. `aedat.py` reads AEDAT-2.0 with
128 x 128 coordinates. The source URL recorded by the loader is
`http://www2.imse-cnm.csic.es/caviar/POKER_DVS/cards_1.aedat`.

## SL-Animals-DVS

Keep each AEDAT recording beside its matching CSV. CSV start/end boundaries are
event indices, not timestamps. The loader slices in file order before filtering
and sorting; changing that order changes which events belong to a sign.

## DVS-Lip

Use `tonic.datasets.DVSLip(save_to="data")`. Released samples are 128 x 128 mouth
crops, not the original full-sensor recordings. The crop builder samples word
classes within the supplied train/test split. Its generated subject column is a
placeholder, so use final-epoch mode (`--val-subjects 0`) rather than pretending
that column defines a subject-disjoint validation split.

# Technical review notes

Areas that deserve review before making new scientific claims:

- `build_crops.process` clears `fovea_id` between selection rules and after the
  last rule. Those `+hyst` calls therefore do not retain a separate previous
  component for each rule across frames. Do not assume their hysteresis matches
  a standalone `PersistentGMM` run. Changing this requires rerunning comparisons.
- The classifier's no-validation mode retains a maximum-test diagnostic. Its
  downstream difference tables use that diagnostic; use final-epoch results for
  unbiased reporting in that mode. DVS-Lip has placeholder subject IDs, making
  the default subject-validation split unsuitable.
- Warm-started GMM indices are not guaranteed physical-object identities.
- `MembraneSurface` decays once per frame and adds current event counts without
  intra-frame decay. It is an approximation to event-time integration.
- `max_iou_quad` uses a finite translation search; it estimates the ceiling.
- Several functions assume nonempty sorted timestamps and sufficient events.
  Frame iteration omits under-populated windows and excludes an event exactly
  at the final window-start boundary when that boundary equals the maximum
  timestamp. Boundary conventions should be tested before changing them.
- `success_curve` uses strict threshold comparisons and the mean of sampled
  success rates. It is not continuous numerical integration of the curve.
- Several analysis scripts suppress warnings globally. Inspect convergence
  warnings when evaluating new datasets.
- `load_gesture` modifies a private Tonic file-presence check. Dependency upgrades
  need a dataset-loading regression check.

Historical records include limitations and corrections; consult later entries
when an earlier interpretation has been revised.

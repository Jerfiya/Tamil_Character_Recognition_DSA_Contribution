# Plan — Tamil handwritten character recognition with few-shot deep learning

Written before any code. Hardware found: Intel i5-1335U (12 logical cores), 16 GB RAM,
**no CUDA GPU** (Intel Iris Xe only). Everything is therefore sized for CPU; see
`reports/decisions.md` for every scaling decision.

## Order of work
1. **Audit** (`src/dataset_audit.py`): recursive scan of the parent folder, locate the dataset
   tree dynamically, discover classes, flag empties / nesting / zips / corrupt files / spelling
   variants / missing grid letters, perceptual-hash duplicates, source groups, K+Q eligibility.
2. **Prepare** (`src/prepare_data.py`): dedupe -> `manifest.csv` -> shared preprocessing ->
   cached `images_<size>.npy` -> Protocol A image split and Protocol B class split.
3. **Tests** (`pytest`) then **2-epoch smoke runs** of every method.
4. **Epoch check**: LR finder, LR comparison, 80-epoch ProtoNet budget study, ablations — all
   judged on validation only.
5. **Final training** of every method on Protocol A and Protocol B (3 seeds for the best).
6. **Test evaluation** once per run (guarded by a marker file), all result files stored.
7. **Visualisation** from stored results only, plus `reports/dashboard.html`.
8. **ONNX export**, registry / predict / canvas / webcam apps, enrol a missing letter.
9. **Summary**.

## Design choices that keep this feasible on a laptop CPU
- All images are preprocessed once to 64x64 uint8 and held in memory; augmentation is batched
  tensor code (no per-image Python loops).
- An "epoch" is a fixed number of episodes (config) so curves are comparable across methods.
- Validation embeds each val image once per epoch, then scores a FIXED list of episodes on the
  cached features (deterministic, no augmentation).
- Test evaluation embeds each test image once per checkpoint and runs the full
  N-way x K-shot grid (>= 600 episodes per cell) on cached features.
- Several training runs execute in parallel, each limited to a few CPU threads.

## Leakage controls
- Near-duplicates removed before any split; cross-class identical images dropped as label conflicts.
- Protocol B classes are disjoint between base / val / novel.
- Support and query images never overlap inside an episode (sampler invariant + unit test).
- Test data is read only by `evaluate.py`, which refuses to run twice on the same run.

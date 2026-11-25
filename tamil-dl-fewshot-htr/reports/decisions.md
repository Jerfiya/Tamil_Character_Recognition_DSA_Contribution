# Decisions log

Every non-obvious choice, why it was made, and what it was based on. Choices about epochs, learning
rate, architecture, loss, thresholds and augmentation use **validation data only**. Test data is read
only by `src/evaluate.py`, which marks each checkpoint in `test/DONE.json` and refuses a second look.

## Hardware
- **No CUDA GPU** (Intel i5-1335U, 12 logical cores, 16 GB RAM, Intel Iris Xe). PyTorch 2.15 CPU build.
  AMP is wired in (`train.amp: auto`) but only activates on CUDA. For a GPU, install the CUDA wheel of
  torch (see README; WSL2 + CUDA also works) and raise `train.episodes_per_epoch` to 200.
- Several runs are trained in parallel, each with `threads: 2` (CPU throughput scales poorly past 2
  threads per process for these small convnets). A wall-clock epoch is 2-5 minutes depending on load.
- Scaled for CPU: one epoch = **50 episodes** (not 200) of 20-way 5-shot 5-query; Conv4 (64 ch) is the
  default backbone; ResNet widths are 1/4 of the usual (16 base channels); final runs use 15 epochs
  (supported by the epoch study, see `reports/epoch_study.md`); ablations use 8 epochs.

## Data (from the audit, `reports/audit.md`)
- Dataset found by recursive scan at `..\tamil_handwritten_Dataset` (never hard-coded). It is treated as
  read-only; a content fingerprint (`data/interim/dataset_fingerprint.json`) is recorded before any
  processing and checked at the end with `python src/check_dataset_untouched.py`.
- 247 classes with images, 14,463 PNG files, 0 corrupt, 0 zips, 0 nested or empty class folders in the
  current tree. All of these cases are still handled in code and covered by `tests/test_data_discovery.py`.
- **Every source image is 48 x 48** (single distinct size). Input size is therefore **48** by default;
  64 and 96 caches exist for the input-size ablation and the pretrained backbones (upsampling cannot
  add information, so 128 was cached but not trained on CPU).
- **Soft thresholding** (`preprocess.soft: true`): Otsu gives the ink mask, but grey levels inside the
  dilated mask are kept. At 48 px, hard binarisation broke thin strokes in the sample grid.
- Near-duplicates: 64-bit pHash, Hamming <= 4 within a class -> 258 removed **before any split**
  (table of counts at thresholds 0-10 in the audit; 4 sits at the knee). Cross-class identical images
  would be dropped as label conflicts (0 found). 3 blank images were dropped after preprocessing
  (`data/processed/blank_dropped.csv`).
- **தெள / லெள / ளெள** are spelled consonant + ெ + ள, while the other 15 rows use the single ௌ sign. They
  are **kept as distinct classes, not merged or renamed**, and flagged as "possible spelling variants of
  தௌ / லௌ / ளௌ". **Please review:** if they are the same letter, rename the folders yourself; the
  pipeline will pick that up on the next `prepare_data.py`.
- Missing from the 18 x 12 grid: **தௌ, லௌ, ளௌ** (which may be exactly the three variants above).
  Listed as "enrollable later"; தௌ is enrolled without retraining in the deployment demo.
- 8 classes have < 15 images after dedupe (ணே, ணெ, ழா, ழ: 4; ணு, ணூ: 5; ழி, ழீ: 6). They are forced
  into the Protocol B *base* part (cannot form evaluation episodes) and in Protocol A get at least one
  val and one test image when they have >= 3 images.
- Source groups come from file names: the largest group is "other" (78%, person names such as
  `Abhinaya.T_IT.jpg`). A coarser **writer key** (name stem) was also derived. Protocol A splits are
  stratified by source group, so every pool has every source in proportion; `source_aware.csv` then
  tests support and query drawn from *different* source groups.
- Class balance: training episodes sample classes uniformly (balanced by construction); the supervised
  baseline uses `ClassBalancedBatchSampler`.

## Protocols
- **A (known classes):** per class, 70/15/15 image split (`data/processed/splits_A.csv`). Support from
  the train pool, queries from the val pool (model selection) or the test pool (final evaluation).
- **B (novel classes):** classes split 70/15/15 by category -> 173 base / 37 val / 37 novel; 31 of the
  novel classes are **hard-novel** (consonant and vowel sign both seen among base compounds).
  `data/processed/class_split.json`. Note that லெள (a flagged variant) fell into *novel* and தெள into *val*.
- Validation uses a **fixed list of episodes** (seed 1234, no augmentation): 100 x 10-way 5-shot 5-query
  per epoch, so val loss is comparable from epoch to epoch and between methods.

## Methods
- **ProtoNet** (main): squared Euclidean / embedding dim x 16 x learnable temperature (log-param). The
  division by D keeps the initial logit scale independent of `emb_dim`, so the emb-dim ablation compares
  geometry rather than logit scale.
- Embedder head: 3x3 (Conv4) or 2x2 adaptive pooling before the projection. Global average pooling throws
  away *where* a vowel sign sits, which is what separates கெ, கே and கை.
- **Siamese/triplet: fixed in this session.** Pure batch-hard triplet mining (hardest positive vs hardest
  negative per anchor) **collapsed**: all embeddings went to one point and the loss stayed at the margin
  (0.30) for 8 epochs, val accuracy 17% on 10-way (`runs_failed/…_siamese_conv4_B_final`). Replaced by
  "batch-all, active-only" triplet loss (mean over margin-violating triplets, the hard-negative mining).
  This version learns from epoch 1.
- **FO-MAML: fixed in this session.** Validation crashed ("differentiated tensors do not require grad")
  because the EMA copy's parameters have `requires_grad=False`; the inner loop now builds differentiable
  leaves from them. A second MAML bug: eval-mode BatchNorm used running statistics gathered under
  constantly changing fast weights, which put validation at chance (11% on 10-way, while the same checkpoint
  scored 37-42% with batch statistics). MAML now always normalises with the current batch's statistics,
  the standard MAML practice, and the run was restarted (`runs_failed/` keeps the broken attempts).
  MAML val/test use 30 / 100 episodes and a reduced grid (it adapts on raw images per
  episode and is ~20x slower on CPU).
- **ArcFace -> episodic**: ArcFace (s = 24, m = 0.35) on the 173 base classes, then a ProtoNet initialised
  from its EMA backbone (`method.init_from`), lr 3e-4, 1 warm-up epoch, same 15-epoch budget.
- **Baseline++**: cosine-classifier pretraining on base classes; at test time the backbone is frozen and a
  cosine head is fitted on the K support images of each episode (60 steps, lr 0.05).
- **Transductive refinement**: 5 soft k-means steps on the episode's unlabeled queries, reported in
  separate rows (`transductive.csv`), never mixed into the main numbers.
- **Supervised CE**: Protocol A only (it needs every class at training time). Its val loss is CE on the
  val pool; at test time it is evaluated like the others (prototypes from its embedding).

## Training recipe
- AdamW (wd 5e-4), linear warm-up (2 epochs) + cosine decay to 1% of the peak, grad clipping at 5, EMA of
  weights (decay 0.995; **the EMA model is validated, checkpointed and tested**), label smoothing 0.05 on
  the episode CE. Early stopping on **val loss** (patience 10), overfitting detector (train falling and
  val rising for 3 epochs), plateau detector (train improves < 0.002 over 6 epochs).
- The clipped-step fraction is 1.0 for ArcFace and Baseline++ (their gradient norm is 7-11, above the clip
  of 5) and ~0 for ProtoNet. This is logged as a diagnostic; it acts as a fixed-norm update for these
  heads and was not changed, so that every method keeps the same recipe.
- Augmentation (train episodes only): rotation ±10°, shift/zoom ±10%, shear ±8°, elastic deformation,
  Gaussian noise/blur, stroke thickness (dilate/erode), brightness/contrast, small random erasing.
  **No flips, no 90/180/270 rotations, no CutMix.** Train accuracy is computed on augmented episodes, so
  train loss sits above val loss. A negative generalisation gap is expected here and does not mean the
  model is underfitting.

## Calibration and open set
- Temperature fitted on the **val** full-set problem (coarse-to-fine search on log T; LBFGS stopped early
  on nearly flat logits). ECE uses 15 bins.
- Open-set score = max logit against enrolled prototypes. Threshold = value that accepts 95% of enrolled
  **val** queries (half the val classes enrolled, half unseen). The test only reports the result.

## Results-driven choices (filled in from validation results)
See `reports/epoch_study.md` and `reports/ablations.md`. The summary of the final choices is appended
below once those runs complete.

## Session 3 (2026-11-17): interruption and the reduced study plan
- The session-2 batches were killed at about 23:02. Partial runs were resumed exactly from `last.pt`
  (Baseline++ at epoch 12, Siamese at 7, epoch study at 7). Runs that never logged an epoch were moved to
  `runs_failed/interrupted_empty/` and restarted.
- At about 02:30 the laptop ran critically low on memory (16 GB) while three 96-px ImageNet-backbone ablations
  started at once. EfficientNet-B0 at 96 px alone used about 3.6 GB. The launcher was stopped.
- **The user chose the faster plan.** Kept: the LR comparison (3e-3, 1e-3, 3e-4, 1e-4), MobileNetV3-L
  pretrained vs scratch at 96 px, ArcFace-pretrain vs none (the `arcft_final` run vs ProtoNet final), and
  3 ProtoNet seeds. **Dropped for time:** the train-way, train-shot, embedding-dim, distance, EMA, input-size,
  ResNet-12, ResNet-18 and EfficientNet-B0 ablations. The EfficientNet-B0 run was stopped at 1/8 epochs and
  archived in `runs_failed/aborted_for_time/`. These remain untested, and `run_batch3.sh` still lists them,
  so they can be run later.
- Test evaluations run one at a time (`eval_queue.sh`) to keep memory in check.

## Results-driven choices (validation only; filled in 2026-11-17)
- **Epoch budget** (40-epoch study, Protocol A, early stopping off): val loss bottoms out at epoch 36
  (0.2215; smoothed minimum 37) with no overfitting flags. The 15-epoch final runs therefore **under-train**.
  The recommended budget is ~44 epochs with early stopping. Session 2 picked 15 from only the first 7 epochs.
  That choice is reversed here, but the final runs could not be retrained at 44 epochs in the time available.
  Their numbers are lower bounds.
- **Learning rate:** 3e-3 > 1e-3 > 3e-4 > 1e-4 by best val loss after 8 epochs (0.306 / 0.338 / 0.419 / 0.564).
  The final runs used 1e-3 (chosen from the LR finder's steepest-descent point, 1.4e-3). For the next
  training round, use 3e-3.
- **Pretrained vs scratch:** MobileNetV3-L ImageNet-pretrained at 96 px reaches val loss 0.2204 / 93.98% in
  8 epochs. From scratch it reaches 0.7066 / 75.22%, and Conv4 reaches 0.3383 / 89.46%. Pretraining is the largest single effect measured.
- **ArcFace pretrain -> ProtoNet fine-tune** is the best Conv4 recipe (val 0.2605 / 92.68% vs 0.2776 / 91.94%
  for ProtoNet from scratch). The margin is inside the seed spread (ProtoNet seeds: 91.6-92.6%).
- **Deployed model:** the run with the lowest validation loss overall, `abl_mnv3_pre96` (ProtoNet,
  MobileNetV3-L pretrained, 96 px). It was selected on val loss before its test results were read.

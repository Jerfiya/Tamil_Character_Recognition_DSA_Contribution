# Model comparison (TEST set)

One row per run; numbers from the PRIMARY checkpoint (`best_val_loss.pt`, chosen on validation loss). Accuracy = mean over >= 600 test episodes (± 95% CI) on the run's own test split: Protocol A = known classes (support from train images, queries from held-out test images); Protocol B = 37 novel classes never seen in training. *Full-set* = every test query classified among all classes of the split with 5 support images each. Latency = CPU, batch 1, embedding only.

| method | backbone | proto | tag | params | CPU ms | best ep / run | best val loss | 5w1s % | 5w5s % (±CI) | 20w5s % | all-way 5s % | hard-novel 5w5s % | full-set % | macro F1 | ECE before → after | open-set AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Matching Net | conv4 | B | final | 186k | 1.3 | 15 / 15 | 0.2756 | 88.3 | 95.0 ± 0.4 | 86.3 | 80.6 | 94.7 | 80.7 | 0.786 | 0.166 → 0.032 | 0.712 |
| ProtoNet | conv4 | B | final_seed7 | 186k | 1.3 | 15 / 15 | 0.2641 | 87.5 | 94.7 ± 0.5 | 85.5 | 79.9 | 94.6 | 79.4 | 0.788 | 0.130 → 0.045 | 0.923 |
| ProtoNet | conv4 | B | final_seed123 | 186k | 1.2 | 15 / 15 | 0.2857 | 87.7 | 94.6 ± 0.4 | 85.8 | 80.4 | 94.8 | 80.6 | 0.794 | 0.139 → 0.048 | 0.914 |
| ProtoNet | conv4 | B | arcft_final | 186k | 1.2 | 15 / 15 | 0.2605 | 87.2 | 94.6 ± 0.5 | 86.4 | 81.2 | 94.7 | 78.9 | 0.783 | 0.107 → 0.046 | 0.924 |
| ProtoNet | conv4 | B | final | 186k | 1.2 | 15 / 15 | 0.2776 | 87.5 | 94.4 ± 0.5 | 85.8 | 80.3 | 94.5 | 79.2 | 0.777 | 0.127 → 0.042 | 0.919 |
| Siamese (triplet) | conv4 | B | final | 186k | 1.3 | 14 / 15 | 0.2195 | 89.0 | 94.1 ± 0.5 | 83.6 | 76.4 | 94.0 | 77.1 | 0.759 | 0.028 → 0.026 | 0.914 |
| Relation Net | conv4 | B | final | 313k | 1.2 | 14 / 15 | 0.3028 | 86.8 | 94.0 ± 0.5 | 83.2 | 76.3 | 93.6 | 77.4 | 0.750 | 0.052 → 0.029 | 0.871 |
| ArcFace (cosine protos) | conv4 | B | final | 208k | 1.2 | 15 / 15 | 0.4355 | 84.9 | 93.2 ± 0.5 | 82.9 | 76.8 | 92.5 | 76.7 | 0.750 | 0.046 → 0.039 | 0.897 |
| Baseline++ | conv4 | B | final | 208k | 1.2 | 15 / 15 | 0.5019 | 84.3 | 92.4 ± 0.5 | 81.3 | 75.4 | 91.8 | 74.6 | 0.734 | 0.072 → 0.040 | 0.725 |
| FO-MAML | conv4 | B | final | 186k | 1.4 | 8 / 8 | 0.6818 | 86.1 | 71.3 ± 5.4 | 80.7 | — | 71.4 | 67.4 | 0.622 | 0.415 → 0.049 | 0.740 |
| ProtoNet | conv4 | A | final | 186k | 1.2 | 15 / 15 | 0.2791 | 90.4 | 95.3 ± 0.4 | 89.1 | 60.9 | — | 61.6 | 0.593 | 0.220 → 0.065 | 0.758 |
| Supervised CE | conv4 | A | final | 217k | 1.2 | 15 / 15 | 1.6007 | 87.3 | 93.5 ± 0.5 | 85.2 | 54.4 | — | 56.1 | 0.540 | 0.384 → 0.091 | 0.691 |
| ProtoNet | mobilenetv3_large | B | abl_mnv3_pre96 | 4325k | 4.9 | 8 / 8 | 0.2204 | 89.7 | 95.6 ± 0.4 | 88.0 | 83.5 | 95.3 | 83.3 | 0.820 | 0.164 → 0.044 | 0.938 |
| ProtoNet | conv4 | B | lr3e-3 | 186k | 1.3 | 8 / 8 | 0.3063 | 86.4 | 94.1 ± 0.5 | 83.8 | 77.6 | 94.0 | 77.1 | 0.761 | 0.103 → 0.043 | 0.907 |
| ProtoNet | conv4 | B | lr1e-3 | 186k | 1.4 | 8 / 8 | 0.3383 | 85.3 | 93.6 ± 0.5 | 83.1 | 76.4 | 93.6 | 75.5 | 0.743 | 0.100 → 0.041 | 0.900 |
| ProtoNet | conv4 | B | lr3e-4 | 186k | 1.4 | 8 / 8 | 0.4193 | 81.7 | 92.0 ± 0.6 | 79.0 | 71.4 | 91.6 | 69.4 | 0.682 | 0.076 → 0.031 | 0.861 |
| ProtoNet | conv4 | B | lr1e-4 | 186k | 1.4 | 8 / 8 | 0.5642 | 77.6 | 89.8 ± 0.6 | 73.6 | 64.7 | 88.9 | 64.3 | 0.625 | 0.072 → 0.035 | 0.828 |
| ProtoNet | mobilenetv3_large | B | abl_mnv3_scr96 | 4325k | 5.5 | 8 / 8 | 0.7066 | 75.3 | 85.0 ± 0.8 | 62.7 | 50.4 | 85.0 | 52.6 | 0.496 | 0.160 → 0.060 | 0.801 |
| ProtoNet | conv4 | A | epochstudy | 186k | 1.2 | 36 / 40 | 0.2216 | 92.8 | 96.6 ± 0.3 | 91.8 | 68.8 | — | 69.9 | 0.675 | 0.227 → 0.064 | 0.810 |

## ProtoNet (Protocol B) across seeds — 5-way

| split | K | mean % | std % | seeds |
|---|---|---|---|---|
| B_hard_novel | 1 | 87.7 | 0.3 | 3 |
| B_hard_novel | 5 | 94.6 | 0.2 | 3 |
| B_novel | 1 | 87.6 | 0.1 | 3 |
| B_novel | 5 | 94.4 | 0.3 | 3 |

## Checkpoint transparency (5-way 5-shot test accuracy)

| run | best_val_loss.pt (epoch) | best_val_acc.pt (epoch) | last.pt (epoch) |
|---|---|---|---|
| 20261116_174746_matchingnet_conv4_B_final | 95.0 (15) | 95.0 (15) | 95.0 (15) |
| 20261117_010748_protonet_conv4_B_final_seed7 | 94.7 (15) | 94.6 (14) | 94.7 (15) |
| 20261117_010748_protonet_conv4_B_final_seed123 | 94.6 (15) | 94.6 (14) | 94.6 (15) |
| 20261117_010748_protonet_conv4_B_arcft_final | 94.6 (15) | 94.6 (13) | 94.6 (15) |
| 20261116_174746_protonet_conv4_B_final | 94.4 (15) | 94.4 (15) | 94.4 (15) |
| 20261116_222957_siamese_conv4_B_final | 94.1 (14) | 94.3 (15) | 94.3 (15) |
| 20261116_174746_relationnet_conv4_B_final | 94.0 (14) | 94.1 (15) | 94.1 (15) |
| 20261116_185222_arcface_conv4_B_final | 93.2 (15) | 93.2 (15) | 93.2 (15) |
| 20261116_185539_baselinepp_conv4_B_final | 92.4 (15) | 92.4 (15) | 92.4 (15) |
| 20261117_012947_maml_conv4_B_final | 71.3 (8) | 71.3 (8) | 71.3 (8) |
| 20261116_174747_protonet_conv4_A_final | 95.3 (15) | 95.3 (15) | 95.3 (15) |
| 20261116_174747_supervised_conv4_A_final | 93.5 (15) | 93.5 (15) | 93.5 (15) |
| 20261117_020113_protonet_mobilenetv3_large_B_abl_mnv3_pre96 | 95.6 (8) | 95.6 (8) | 95.6 (8) |
| 20261117_022705_protonet_conv4_B_lr3e-3 | 94.1 (8) | 94.1 (8) | 94.1 (8) |
| 20261117_024929_protonet_conv4_B_lr1e-3 | 93.6 (8) | 93.6 (8) | 93.6 (8) |
| 20261117_025007_protonet_conv4_B_lr3e-4 | 92.0 (8) | 92.0 (8) | 92.0 (8) |
| 20261117_030315_protonet_conv4_B_lr1e-4 | 89.8 (8) | 89.8 (8) | 89.8 (8) |
| 20261117_022702_protonet_mobilenetv3_large_B_abl_mnv3_scr96 | 85.0 (8) | 85.0 (8) | 85.0 (8) |
| 20261116_222957_protonet_conv4_A_epochstudy | 96.6 (36) | 96.6 (39) | 96.6 (40) |

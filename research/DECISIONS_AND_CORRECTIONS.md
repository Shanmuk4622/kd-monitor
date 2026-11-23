# Decisions, mistakes and improvements

## Parallel execution correction

Earlier instructions required the researcher to enter a repository and version numbers and used a single writer. The revised notebooks prefill the repository, use content-based input identity and fence each writer independently. Teacher fits and cached data have one owner each.

**23 November 2026, runnable delivery.** Replaced the binary-only assumption in the new notebooks with checked Python-batch imports. Kept the old notebook self-contained as an earlier release. Used one embedded code snapshot across the three stages so fresh sessions do not need additional source uploads. The preflight saves a two-update remote recovery anchor. The training notebook downloads that anchor and compares the completed trajectory with a fresh reference before any measured fit. A status heartbeat is distinct from a verified resumable fit checkpoint.

**23 November 2026, before benchmark training.** The first runner assumed the binary CIFAR-100 archive. The requested Kaggle CIFAR sources are named as Python distributions. The [current plan](KAGGLE_EXPERIMENT_PLAN.md) calls for a new input adapter, data identity and campaign prefix. It adds 18 MobileNetV4 fits to test both student families on every dataset. The comparison thresholds and student exposure budget carry forward. Kaggle inventory and live checkpoint recovery are pending.

## Decisions made before the first measured fit

| Date | Decision | Reason and consequence |
| --- | --- | --- |
| 22 Nov 2026 | Use CIFAR-100 for one primary question, then vary the model or dataset one at a time. | A large leaderboard would hide why a student improved. CIFAR-100 offers 100 classes and a manageable first target. The transfer set remains incomplete as a full architecture-by-dataset factorial. |
| 22 Nov 2026 | Use a fixed average of four views and a two/six allocation. | It leaves two paid pilot views for a within-image comparison, ensures equal per-class quotas, and makes every slot's exposure count an integer over 60 epochs. The optimal cache size is outside the claim. |
| 22 Nov 2026 | Distinguish the shuffled, uniform and entropy controls. | The shuffled policy isolates who receives extra views. Uniform checks whether unequal allocation helps. Entropy tests a simpler explanation for any gain. |
| 22 Nov 2026 | Pair five independently redrawn blocks and use one teacher per dataset. | Pairing controls student initialization and augmentation randomness. Teacher-seed variability remains open. Five blocks are a compute choice and do not guarantee precision for a small effect. |
| 22 Nov 2026 | Use a +0.30-point useful gain and a −0.50-point quality margin for the separate energy comparison. | These are prospective research tolerances selected for this design. They are not thresholds reported by the supplied papers. |
| 22 Nov 2026 | Retain final-epoch student accuracy as the primary endpoint. | A validation-best epoch can differ between arms. A fixed endpoint ensures exactly the same number of updates and source-image exposures. |
| 22 Nov 2026 | Prefer dense FP32 logits and native-resolution crops. | This prevents label quantization and resized small images from becoming extra explanations. The conclusion will be specific to this augmentation family. |
| 22 Nov 2026 | Treat existing tyre and early-exit material as project context only. | The attached handoff's Kaggle, evidence and recovery rules transfer. Its data, old results, repository paths and completed model identities belong to different experiments. |

## Mistakes found and corrections made

| Observation | Correction and check |
| --- | --- |
| The local project directory contained two PDFs. A broad search for experiment guidance returned other research projects. | Resolved the two exact Markdown filenames, read the handoff and empty brief, and filled a new copy for this project. Both input identities are in the root README. |
| The first PDF extraction used the Windows console's default encoding and stopped at a mathematical character. | Read the local PDFs with Unicode output, and visually checked Table 2 of the AIoT paper and Table 4 of the PLOS paper. Kept reported quantities tied to their hardware and comparator. |
| An initial design that compared live KD with a saved bank changed view diversity at the same time as delivery. | Added an exact same-pixel logit/loss/gradient check, and made the primary scientific contrast between two finite banks of equal cost. |
| Charging only extra allocation queries would make the ranking procedure appear cheaper than it is. | Include both probes in each standalone budget. The fixture counts actual build forwards, and records extra parity-audit forwards separately. Store physical shared work and standalone-equivalent counts as different fields. |
| An early environment check imported torchvision after torch and encountered a Windows imaging DLL load error. | Verified the primary ResNets after adding the environment's DLL directory and importing Pillow before torchvision. The portable cache fixture uses tensor crops and has no Pillow dependency. No packages were replaced. |
| The local environment lacks timm, and the original CIFAR-100 binary was unavailable in the project. | The MobileNetV4 check and official-data audit remain pending. Verified the source architecture identity, made the CIFAR input explicit and tested role/duplicate handling with a constructed fixture. |

## Improvements to test or implement

1. Qualify the measured writer and full-update restart on dual T4 with a verified source snapshot, a recorded peak-space forecast and a real cross-session restore.
2. Run the implemented independent-view check. It samples ten fitting images per class and evaluates two fresh views outside all six candidate slots. Its 2,000 teacher queries per guided block are charged to diagnostics.
3. If a causal claim about the soft targets becomes important, add a separately versioned pair of label-only fits on the identical guided and shuffled banks. The current primary comparison measures the whole view-allocation policy.
4. If a five-block interval spans harm and useful gain, disclose the range. A future precision study needs a prospective seed budget and a new analysis record.

For subsequent work append the date, old rule, new rule, trigger, affected artifacts and whether any validation or final outcomes had already been seen. Preserve failed run identities and the original protocol.

## Implementation corrections on 23 November 2026

| Issue identified | Change | Verification |
| --- | --- | --- |
| Recording only the live student's logits would prevent an independent calculation of its distillation loss. | Store the teacher's raw FP32 outputs for each live training view and link them to the same rows as the student's outputs. | Recomputed the objective from the saved arrays in every fixture arm on both devices. |
| Saving model weights without the matching optimizer and evidence cursor would permit a duplicated update after interruption. | Alternate complete rolling states. Select the new generation only after writing its evidence. | Injected a pre-selection crash and verified identical predictions, optimizer and random-number states in fresh processes. |
| A short timing probe omits validation, epoch weights and most evidence writes. | The target capacity check now runs one complete disposable epoch at the scientific batch size for each architecture. It reports observed bytes by file family and separate compute-only and local elapsed-time forecasts. | Verified the path with both actual ResNets on 32 × 32 synthetic images. Target measurements will be recorded on the first real Kaggle execution. |
| The local NVML index can differ from a CUDA device's logical index. | Identify the active board by UUID. Record both reserved boards and preserve missing measurements. | Checked aggregation, missingness and energy units with analytical fixtures. The first target trace will establish sensor availability. |
| A stale local checkpoint could be mistaken for a completed remote fit awaiting audit. | Recover the complete remote selection when the hashes of the local and remote histories differ at that stage. Bind each final audit to the exact selected fit. | Checked older-revision and remote-ahead recovery offline. The live cross-session check is pending. |

These engineering changes preceded all official-data training and evaluation. The [scientific protocol](PROTOCOL.md) and primary thresholds retain their original version.

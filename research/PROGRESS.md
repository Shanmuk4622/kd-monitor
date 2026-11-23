# Progress, validation receipts and open work

## Four-worker release, 23 November 2026

All six campaign notebooks now default to [shanmuk4622/tinyimagenet200](https://www.kaggle.com/datasets/shanmuk4622/tinyimagenet200). Kaggle Data Explorer shows one top-level dataset folder. The full local extracted copy passed the updated loader, including the final-label checks: 200 classes, 100,000 training images, 10,000 official validation images, 90,000 fitting examples, 9,996 cleaned development examples and 9,993 cleaned final examples. Local input identity was `fe0d02f4eb3099b12931d7d4c1cdf708299bb817a3b5f0c4d52c57318d5fd49e`. The Kaggle session must still verify its mounted bytes and run the three GPU capacity probes. The previous Kaggle run confirmed both CIFAR qualifications.

**23 November 2026, notebook delivery.** Implemented the Kaggle CIFAR-10/CIFAR-100 Python-batch and Tiny ImageNet input paths, both student architectures, resumable group scheduling and sealed final review for 75 students. The three [Kaggle notebooks](../notebooks/KAGGLE_RUN_INSTRUCTIONS.md) share one embedded runtime. Local checks used generated inputs: both CIFAR formats at their expected counts, a small Tiny ImageNet ZIP, all six residual-model output shapes, deterministic six-arm recovery and a simulated 75-fit final release yielding 150 cohort rows. The local environment lacked timm. Notebook 01 checks the real mobile model at all resolutions and class counts before long training. Actual Kaggle mounts, dual-T4 performance, Hub storage and live transfer are pending user execution.

**23 November 2026, campaign amendment.** Prepared the [Kaggle-only experiment and retention plan](KAGGLE_EXPERIMENT_PLAN.md) and [expanded matrix](RUN_MATRIX_V2.csv). The earlier 57-student design is the baseline snapshot. The current target is 75 students and three teachers. Kaggle mount versions and file layouts await inspection. The existing primary notebook expects binary CIFAR-100. Measured execution awaits the revised adapters and on-target recovery qualification.

## Status on 23 November 2026

| Stage | State | Evidence |
| --- | --- | --- |
| Two supplied PDFs and Markdown instructions | Reviewed | [Evidence register](LITERATURE.md) and [filled brief](EXPERIMENT_BRIEF.md) |
| Additional prior art and paper question | Reviewed and specified | FKD, consistent teaching, dataset reinforcement, FerKD, adaptive sampling and class coverage mapped to explicit controls |
| Preregistered-style design | Specified locally | [Experimental protocol](PROTOCOL.md). Archive the final source/version before measured execution. No external preregistration is claimed. |
| Run-by-run plan | Enumerated | [60-row run matrix](RUN_MATRIX.csv), including dependencies, independent cache builds, later reuse and planned update counts |
| First notebook | Implemented, executed locally | [Source notebook](../notebooks/00_cache_preflight.ipynb) and [executed copy](../evidence/local_preflight/executed_preflight.ipynb) |
| Synthetic allocation, live/cache equivalence and split logic | Passed on local inputs | [Machine-readable fixture receipt](../evidence/local_preflight/fixture/receipt.json) |
| Actual CIFAR-100 split | Writer implemented, original archive pending | Expected binary digest in [run guide](RUN_GUIDE.md). Exact post-audit validation and test denominators are unknown. |
| Small-image residual models | Local forward shapes checked with random weights | Counts below. Trained checkpoints pending. |
| CIFAR-100 primary trainer, audit and restart | Implemented, exercised on synthetic data | [Training notebook](../notebooks/01_cifar100_master.ipynb), [executed copy](../evidence/primary_master/executed_primary_master.ipynb) and [operating instructions](PRIMARY_MASTER.md) |
| MobileNetV4, CIFAR-10 and Tiny ImageNet transfers | Specified in the run matrix | These stages need dataset- and architecture-specific execution. Local timm is absent. |
| Dual T4, live Hub restore and campaign budget | Pending | No occupied remote session, confirmed destination, sensor trace, measured T4 throughput or storage projection in this version. |
| Benchmark fits and independent results audit | 0 of 57 planned students, 0 of 3 teachers | No measured accuracy, energy saving, selected model or remote revision. |
| Manuscript | Introduction and methods drafted | [Working manuscript](PAPER_DRAFT.md). Results and conclusions await evidence. |

## Local engineering receipt

The executed notebook ran its five code cells in a fresh local Jupyter kernel. Software was Python 3.11.15, PyTorch 2.5.1+cu121 and NumPy 2.4.6 in cv_conda. Visible accelerators were one NVIDIA GeForce GTX 1650 and the CPU. The T4 environment is a separate qualification target.

The [SHA-256 inventory](../SHA256SUMS.txt) identifies the sources, submitted notebook, executed notebook, protocol and retained fixture files for this design snapshot. Future measured-run receipts must bind the actual benchmark data and checkpoint identities as well.

| Check | CPU | GTX 1650 | Acceptance |
| --- | ---: | ---: | --- |
| Standalone teacher image evaluations per policy | 100 | 100 | Exactly 4 × 25 fitting images, including both pilot views |
| Dense FP32 target bytes per policy | 1,200 | 1,200 | 100 rows × 3 classes × 4 bytes |
| Largest absolute cached vs recomputed logit difference | 0 | 0 | ≤ 0.00001 |
| Largest absolute KD loss difference | 0.000000059605 | 0.000000059605 | ≤ 0.000001 |
| Largest absolute student-gradient difference | 0.000000007451 | 0.000000007451 | ≤ 0.00001 |

All four policies passed the per-class budget, distinct slot and 60-epoch exposure checks, including the odd-sized class with a four-view median. The fixture checks frozen-teacher gradients and reopens the persisted view index and logits before the comparison. Per device, it physically used 50 shared probe evaluations, four sets of 50 extra allocations and 108 independent verification forwards per policy. The 100-call standalone figure describes cache construction for one policy. [CPU policy rows](../evidence/local_preflight/fixture/cpu/policy_checks.csv) · [GPU policy rows](../evidence/local_preflight/fixture/cuda-0/policy_checks.csv)

The separate split fixture deliberately introduced overlap. Eight candidate training records produced four fitting rows, three eligible validation rows and one excluded validation row. Four final candidates produced two eligible rows and two exclusions. It exported zero final labels. These are synthetic counts.

| Residual architecture | Parameters for 10, 100, 200 classes | Tested outputs with batch two |
| --- | --- | --- |
| ResNet-18 | 11,173,962 / 11,220,132 / 11,271,432 | 2 × 10 at 32 px, 2 × 100 at 32 px, 2 × 200 at 64 px |
| ResNet-34 | 21,282,122 / 21,328,292 / 21,379,592 | 2 × 10 at 32 px, 2 × 100 at 32 px, 2 × 200 at 64 px |

## Experiment sequence

The next measured milestone is the CIFAR-100 ResNet-34 teacher and the six seed-42 ResNet-18 fits. The Kaggle master first checks the original archive and split, qualifies each GPU, restores a pinned revision in a fresh process and benchmarks one full disposable epoch per architecture. After all five primary blocks and their audits, the final-review mode can score the test set. The transfer stages address the scope of any resulting claim.

## Primary notebook checks on 23 November 2026

| Check | Observed outcome | Record |
| --- | --- | --- |
| Six student configurations | Each completed nine updates and 72 observations on each of two local devices. This fixture used 24 fitting images and a small CNN. | [CPU](../evidence/primary_master/cpu_qualification.json) and [GTX 1650](../evidence/primary_master/gpu_qualification.json) |
| Actual training evidence | Independent recomputation checked the augmented predictions, cross-entropy, distillation loss, per-epoch source coverage and validation selection in all six arms. Every synthetic cache contained exactly 96 teacher predictions. | [Example exposure table](../evidence/primary_master/live_example/exposure.csv), [optimizer updates](../evidence/primary_master/live_example/updates.csv) and [both raw logit arrays](PRIMARY_MASTER.md#what-to-inspect) |
| Actual primary architectures | A separate CPU check exercised a full disposable epoch for the 100-class ResNet-34 teacher and live-KD ResNet-18 at 32 × 32. Each saw 130 synthetic fitting images in batches of 128 and two, and eight validation images. The storage forecast included epoch and update files. Missing fits and misbound audits both stopped the final evaluation before test access. | [Model and final-release checks](../evidence/primary_master/resnet_and_release_checks.json) |
| Recovery | A crash after writing the next rolling state and before selecting it left the previous checkpoint usable. A fresh process reproduced the final model, optimizer, RNG and all 72 predictions. Both devices rejected a corrupted file and an incompatible run. | [Executed notebook](../evidence/primary_master/executed_primary_master.ipynb) |
| Publication and energy calculations | An offline transport fixture exercised writer conflict, pinned download and remote-ahead recovery. Analytical sensor cases covered cumulative counters, trapezoidal integration and missing energy. | [Separate receipt](../evidence/primary_master/transport_cost_receipt.json) |
| External execution | Official CIFAR-100 runs completed this release: 0. No real dual-T4 forecast, upstream checkpoint revision, test prediction or board-energy result exists yet. | [Inputs for the measured stage](PRIMARY_MASTER.md#measured-training) |

Keep each later entry in this log tied to a run, a file and a verification level. Report the observed outcome, failed checks, and the next unresolved dependency. Use [decisions and corrections](DECISIONS_AND_CORRECTIONS.md) for protocol changes or errors.

The private release at Hugging Face commit edce1cec54ce2a1c5142aa2681d66950837d7a3a contains the six notebooks and the run guide. All seven downloads matched local bytes. The concurrent fixture dispatched 15, 19, 18 and 26 fits to workers 0 through 3, covered all 78 unique fits and assigned the 38 prediction banks to unique owners. Six notebook bootstraps, synthetic CIFAR/Tiny ImageNet ingestion, six-arm deterministic recovery, 75 fixture endpoints and 150 aggregate cohort rows passed local checks. Kaggle preflight and measured training have not run.

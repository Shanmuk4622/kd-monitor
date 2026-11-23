# Experiment brief · allocating teacher views under a fixed budget

Version 1.0, 22 November 2026. This is the filled project copy of the supplied experiment-brief template.

## Request and scope

| Field | Project decision |
| --- | --- |
| Work requested | Literature review, a paper-worthy experimental question, an executable first check, a methods draft and continuing Markdown records. |
| Question | On CIFAR-100, does giving more views to images with high two-view teacher disagreement increase the ResNet-18 student's final top-1 accuracy over giving those same quotas to random images within each class? |
| Observed problem | FKD shows the cost of repeatedly evaluating a teacher. Comparing cached and live KD alone confounds the source of the targets with the number of distinct views. The supplied energy papers measure different cost boundaries. There is no measured allocation result for this project yet. |
| Hypothesis | Two teacher predictions contain a usable signal about the value of paying for additional views of that source image. The chosen allocation will exceed a matched random allocation and will also improve on uniform caching. |
| Falsifier | The upper 95% interval endpoint for the primary gain lies below +0.30 percentage points, ruling out the prespecified useful magnitude at this precision. A clearly negative gain indicates harm. An interval spanning useful gain and zero is inconclusive. |
| Immediate deliverable | The literature review, primary test, staged replications, two executable notebooks, post-run analysis, introduction/methods and verification records. |
| Execution | The [primary master](PRIMARY_MASTER.md) implements the CIFAR-100 teacher and all six student configurations, for one to five prespecified seed blocks. Its offline qualification has run on a CPU and GTX 1650. Official-data fitting and a live checkpoint transfer await the target session and inputs. |

## Comparison and data

| Field | Project decision |
| --- | --- |
| Baseline artifact | Frozen, dataset-specific ResNet-34 chosen by validation over 90 supervised epochs. Primary control uses its predictions with class-shuffled six/two quotas. A uniform four-view bank is the practical comparator. Teacher hashes are assigned after fitting. |
| Variable and invariants | Vary the assignment of distinct views. Match teacher, candidate crop sequence, student initialization, examples per epoch, optimizer updates, KD loss and dense FP32 target budget. |
| Cohorts | CIFAR-100 is primary. CIFAR-10 and Tiny ImageNet are dataset replications. The cross-family check keeps CIFAR-100 and changes only the student architecture. |
| Data separation | Fit all learned quantities and build teacher banks using fitting IDs only. Use exact decoded RGB hashes to remove overlap from validation and the primary final cohort. Also report standard full-split accuracy. |
| Primary endpoint | Per-seed final-epoch clean top-1 percentage on the one frozen, pixel-deduplicated final cohort. Report the mean of five paired percentage-point differences, each seed's counts, and a two-sided 95% paired t interval. |
| Decision and selection | The primary statistical claim needs a positive lower interval bound. A practical recommendation additionally needs a mean gain of at least +0.30 points and a positive lower bound versus uniform under the ordered testing rule. The main student endpoint is epoch 60. No deployment candidate is selected by this study. |
| Secondary endpoints | Entropy-ranked allocation, live teacher, hard labels, per-class recall, NLL, ECE, actual target queries, cache bytes, elapsed time and GPU-board energy. |
| Independence | Blocks redraw the student's initialization, order, candidate views and class-wise shuffle together. All blocks share a dataset and one teacher per dataset. No claim covers the teacher-seed population. |

## Execution and evidence

| Field | Project decision |
| --- | --- |
| Namespace and budget | kd-view-budget/v1. Five primary blocks and three blocks per replication. Maximum 57 student fits and three teachers. The preflight contains zero measured fits. |
| Kaggle setup and hardware | Original read-only inputs. First input is cifar-100-binary.tar.gz. Dual T4 for the eventual measured run. GPU 0 handles timed work and GPU 1 verifies replay outside those windows. Log both devices. No secret is required for the supplied preflight. A measured publisher uses the Kaggle secret HF_TOKEN and a separate KD destination. |
| Space and time | For a CIFAR-100 bank the dense payload is 72 MB. For Tiny ImageNet it is 288 MB. These are lower bounds. Measure CSV, state, epoch weights, cache indexes, reads and peak staging bytes in the actual session. Per-fit runtime is an estimate from observed T4 throughput, recorded before long training. |
| Evidence | Per-image manifests, two probe scores, per-view records, run and update ledgers, augmented predictions, validation/final predictions, per-device samples and publication receipts. Direct CSV for tabular evidence and typed native arrays for dense bodies. See the [run guide](RUN_GUIDE.md). |
| Recovery and publication | One coordinator, complete-update transaction, compatible hashed prefixes, paired control identities, guarded completed stages. Local checkpoints precede 30-minute, stage-end and catchable-stop publication. |
| Stop and promotion | Stop on invalid alignment, leakage, query mismatch, failed continuation or insufficient storage. Missing power readings invalidate the energy claim for the interval. A negative accuracy result is a scientific outcome and remains reportable. |
| Omissions | No facility carbon, deployment energy or wall-plug measurement. Intermediate feature maps and per-parameter historical gradients are outside this logit study. Near-duplicate sources may persist after exact hashing. |

## Completion record

The design, manuscript groundwork and CIFAR-100 implementation are complete. The [progress log](PROGRESS.md) records local checks and the remaining inputs. Benchmark predictions, a measured energy comparison and the paper's conclusions await official-data training.

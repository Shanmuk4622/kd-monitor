# Project context

## Current execution configuration

All six campaign notebooks use [shanmuk4622/tinyimagenet200](https://www.kaggle.com/datasets/shanmuk4622/tinyimagenet200) for Tiny ImageNet.

The campaign uses Shanmuk4622/kd-crop-budget-4w-2026, four fixed worker IDs, GROUP = "all", and SHA-256 input inventories. The run guide in notebooks supersedes earlier scheduling instructions.

Last updated 23 November 2026. This is a proposed image-classification paper about the cost of obtaining supervision from a larger CNN. The working manuscript is **Do Crop-Sensitive Images Deserve More Teacher Queries? A Fixed-Budget Study of CNN Distillation**.

## What the study would establish

A teacher can return a probability for every class on every crop. Those predictions cost time and energy. The experiment asks whether two early predictions can tell us where to spend the next ones. The teacher sees additional crops of the images for which its first two predictions disagree most. Comparing this policy with a random assignment of identical quotas isolates the usefulness of ranking the images. Comparing it with equal quotas tells us whether changing the allocation is worthwhile at all.

| Fixed decision | Project choice |
| --- | --- |
| Main benchmark and endpoint | CIFAR-100, final-epoch ResNet-18 top-1 accuracy, paired over five seed blocks, with a frozen ResNet-34 teacher. |
| Primary resource | 180,000 raw FP32 teacher predictions per cache. Initial probes count. All student configurations receive identical numbers of source exposures and optimizer updates. |
| Secondary resource | GPU-board joules and time for target preparation and subsequent student training. Reuse projections depend on measured costs and the accuracy margin. |
| Transfer boundary | The mobile student changes the architecture on CIFAR-100. CIFAR-10 and Tiny ImageNet change the dataset while retaining the residual pair. |
| Paper state | Research question, literature, methods and analysis rules are written. The directions and magnitudes of the benchmark effects are unknown. |

The [literature review](LITERATURE.md) records what the two supplied PDFs contribute and how the proposed comparison differs from the closest published methods. The [protocol](PROTOCOL.md) fixes the hypotheses, data handling and interpretation. The [run matrix](RUN_MATRIX.csv) is the reference for planned scientific fits. The [manuscript](PAPER_DRAFT.md) contains the current introduction and methods.

## Resuming the work

**Runner delivery, 23 November 2026.** The [Kaggle notebooks and settings](../notebooks/KAGGLE_RUN_INSTRUCTIONS.md) implement the revised 78-fit campaign. The user will run them. NB01 qualified both CIFAR inputs on Kaggle. Tiny ImageNet qualification and the cross-session recovery check are pending.

**Current direction, 23 November 2026.** Use the three specified Kaggle mounts, including the Python CIFAR editions. The [revised experiment plan](KAGGLE_EXPERIMENT_PLAN.md) supersedes the binary-input instructions below for measured work. Both student architectures now run on each of the three datasets. The new [run matrix](RUN_MATRIX_V2.csv) contains 75 students and three teachers.

Both notebooks and the local disposable evidence are in the package. The next scientific run is the CIFAR-100 teacher followed by the six seed-42 students on Kaggle. It requires the original binary archive, a dual-T4 session and a dedicated writable checkpoint destination. An actual target runtime will determine fit time, disk use and energy coverage. The [primary instructions](PRIMARY_MASTER.md) explain how to run it. Append dated observations to [progress](PROGRESS.md) and material changes or failed assumptions to [decisions and corrections](DECISIONS_AND_CORRECTIONS.md).

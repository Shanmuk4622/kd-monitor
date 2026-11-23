# Where should a CNN teacher spend its prediction budget?

**Experiment plan, revision 2 · 23 November 2026 · before benchmark training**

The primary question is whether an image whose teacher predictions change across two crops deserves extra supervision. The extension tests that decision on **every combination of three datasets and two student architectures**. A student receives the same number of updates, sees every fitting image once per epoch and uses the same teacher as its competitors. This makes the assignment of the extra crops interpretable. All dates, seeds, bank identities and acquisition decisions should be recorded before inspecting benchmark outcomes.

## Data come only from the three specified Kaggle inputs

| Input | Expected layout to check at mount | Fitting / validation candidates / final candidates |
| --- | --- | --- |
| [CIFAR-10, shanmuk4622](https://www.kaggle.com/datasets/shanmuk4622/dataset-cifar10-python) | Python distribution, usually five numbered training batches and one test batch, with label names in the metadata. Preserve batch order. | 45,000 / 5,000 / 10,000 at 32 × 32 |
| [CIFAR-100, shanmuk4622](https://www.kaggle.com/datasets/shanmuk4622/dataset-cifar100-python) | Python distribution, usually train, test and meta. Use fine labels for the task. Retain coarse labels for diagnostics. | 45,000 / 5,000 / 10,000 at 32 × 32 |
| [Tiny ImageNet, shanmuk4622](https://www.kaggle.com/datasets/shanmuk4622/tinyimagenet200) | Version 1 shows one `tiny-imagenet-200/` folder with `train/`, `val/`, `test/`, `wnids.txt` and `words.txt`. Read the class list from `wnids.txt` and the final labels from `val/val_annotations.txt` at release. | 90,000 / 10,000 / 10,000 at 64 × 64. The labeled official validation set is the final holdout. |

The counts are design expectations before the overlap audit. Directory names, archive nesting and Kaggle version numbers still need verification on the actual mounts. Attach these three datasets read-only. Disable automatic dataset downloads. Record owner, slug, Kaggle version when exposed by the mount, mounted path, relative filenames, byte counts and SHA-256. Decode to canonical RGB, check shapes, channel order, counts and class mapping, then hash decoded pixels and stable source IDs. Inspect representative decoded images and check class balance. Use class-stratified training-derived validation. Compute normalization on fitting images only. Retain exclusion reasons for exact cross-split matches and repeated evaluation images. Freeze the cleaned and full evaluation cohorts before scoring.

**Compatibility correction.** The released notebook reads the CIFAR-100 binary archive. The two requested CIFAR sources specify the Python edition. The next measured notebook needs Python-batch import for both CIFAR datasets and a Tiny ImageNet loader. Pointing the old notebook at a renamed file would fail its reader and source checks. Create a new campaign namespace, bind it to these inputs and retain the previous synthetic checks as engineering history. No benchmark checkpoints have been produced.

## Scientific comparisons and run budget

Train one native-resolution ResNet-34 teacher per dataset from scratch for 90 epochs, selected on validation. Train ResNet-18 and MobileNetV4-Conv-Small students from scratch for 60 epochs. Use the small-image stems in the original protocol, record feature sizes at each stage and verify the mobile model on 32 × 32 and 64 × 64. Retain SGD, momentum 0.9, batch 128, five-epoch warm-up, cosine decay, FP32, temperature 4 and distillation weight 0.5. Use the same split, teacher and student initialization across competing policies within a model and seed. Pair the source order and candidate transforms, and counterbalance arm order by the seed-based ordering in the matrix.

| Dataset | ResNet-18 student fits | MobileNetV4 student fits | Frozen teacher fits |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 6 methods × 5 seeds = 30 | 3 methods × 3 seeds = 9 | 1 |
| CIFAR-10 | 3 methods × 3 seeds = 9 | 3 methods × 3 seeds = 9 | 1 |
| Tiny ImageNet | 3 methods × 3 seeds = 9 | 3 methods × 3 seeds = 9 | 1 |
| **Total** | **48** | **27** | **3** |

The three shared methods are uniform four crops per image, randomly allocated six/two within class, and six/two assigned by teacher disagreement. Score disagreement by Jensen–Shannon divergence between the teacher's predictions on two crops. Each policy pays for the two initial crops. On each CIFAR dataset the bank budget is 180,000 teacher calls, and on Tiny ImageNet it is 360,000. Amortize an existing bank across architectures only when the dataset, split, seed, teacher weights, crop definitions and raw logits agree. Different models may consume those identical raw image views with the same protocol normalization.

The additional primary controls are hard labels, live distillation on fresh crops, and entropy-ranked allocation. Primary seeds are 42, 123, 2026, 31415 and 27182. Replications use the first three. The resulting 78 fits add 18 mobile runs to the earlier design. They reuse existing banks. Planned work totals 2,090,880 optimizer updates, 38 independently built banks, 8,460,000 bank-building teacher evaluations and 13,500,000 live training-view teacher evaluations. Charge diagnostics, validation, retries and qualification separately. Each run has a row in [RUN_MATRIX_V2.csv](RUN_MATRIX_V2.csv).

## Order of execution

1. **Qualify the three inputs and six student–dataset paths.** Freeze manifests, class maps, model provenance and environment. Run a disposable full epoch for every dataset/model shape and representative live/cached path. Measure peak disk, memory, export bytes and time. Force an interrupted write, publish, open a fresh Kaggle session, fetch a pinned HF revision and compare continuation against uninterrupted disposable training.
2. Launch four independent workers with GROUP set to all. Workers 0, 1 and 2 fit the CIFAR-100, CIFAR-10 and Tiny ImageNet teachers respectively. Worker 3 can start the five CIFAR-100 hard-label controls immediately.
3. Each worker completes its assigned student blocks. Both student architectures in a dataset/seed block reuse the same cached predictions. A consumer waits for the frozen teacher's audit and verifies its selected weights. GPU 0 performs the scientific fit and GPU 1 performs the independent qualification checks. Keep final outcomes sealed.
4. **Release results.** Freeze the complete run list and checkpoints before looking at final outcomes. Evaluate the final-epoch students on both cleaned and standard cohorts. Produce paired predictions, figures and tables from the stored records. Label any unfinished block explicitly.

## What to save during training

Every evidence shard carries dataset, model, method, seed, run ID, source revision, checkpoint generation, row range and SHA-256. Use directly readable CSV for indexes and ledgers, typed native arrays for high-volume tensors, PT for training state and JSON for configurations. Store the raw evidence alongside derived summaries.

| Frequency | Capture | Why keep it |
| --- | --- | --- |
| Once per input, model or bank | Source inventory and hashes, class/split map, exclusions, normalization, environment, code, model definition, initialization digest, seed/order, full FP32 teacher logits, crop/flip keys, image quotas, probe scores and correctness | Rebuild the exact cohort, teacher budget and targets. |
| Every training image exposure | Source ID, label, epoch, update and pre-update weight version, crop/flip, bank slot, prediction, true-label probability and the full pre-update student logits. Save full same-view teacher logits for live KD. Cached runs reference the exact bank row. | Recompute the loss and identify the examples that helped or deteriorated. |
| Every optimizer update | Sample count, summed CE/KL/objective, correct count, LR, gradient norm, step duration, throughput and peak GPU memory. Record nonfinite values, retries and skipped work. | Reconstruct curves with correct denominators and diagnose instability. |
| Every epoch | Weights, optimizer cursor, train/validation counts and losses, per-validation-image logits, predictions and class metrics, current validation-best identity. Record histograms or per-layer weight/gradient norms on a fixed diagnostic batch if space permits. | Revisit learning dynamics and model selection. Use identical diagnostic cadence across methods. |
| During each measured phase | Per-board UUID and timestamp, sampled power or cumulative energy, measurement gaps, warm-up/idle, synchronized timings, memory, query counter, bytes read/written. Keep data preparation, caching, fitting, evaluation, publisher and retry costs identifiable. | Establish cost and break-even under repeated use. Missing readings remain missing. |
| On restart, failure and completion | Full rolling model/buffers, optimizer momentum, scheduler step, RNG states, sampler/crop cursor, partial epoch sums, immutable evidence hashes, local/remote generation, failure log and publication receipt. At release add per-image test logits, predictions, confusion and paired differences. | Restore the same trajectory and audit every reported table. |

Retain every epoch's weights and per-image logits. Full activation and gradient tensors at every step would multiply storage and distort timing. For representation analysis, use epoch weights to regenerate a predetermined, bounded fitting-image probe in evaluation mode. Record that diagnostic's cost separately. Preserve the records of every failed or resumed scientific run.

## Hugging Face publication and recovery

Use one coordinator and a dedicated HF dataset repository. Store its write token in the Kaggle secret named `HF_TOKEN`. The exact destination remains to be supplied. Give the campaign a new prefix, for example `kd-view-budget/kaggle-v2/`, and put immutable shards beneath dataset/model/method/seed. Publish each shard once. A small manifest maps logical filenames to content hashes, durable update ranges and remote revisions.

| Trigger | Required publication |
| --- | --- |
| Every 30 minutes elapsed since the last verified remote save | At the next complete update or cache chunk, flush state and direct evidence, upload pending immutable shards and atomically select their manifest. Keep a timestamp for deadline, actual attempt and verified completion so delays are visible. Long qualification, audit and evaluation stages also need a heartbeat. |
| First input freeze, accepted qualification, selected teacher, each completed cache, each completed fit or audit, and final results | Publish immediately, including the stage receipt and the current complete state. Local epoch files are durable even when the next timed remote commit is pending. |
| Catchable stop, imminent session cutoff or exception | Finish the last complete transaction, flush outstanding evidence and attempt one verified publication. Log a failed transfer locally with retry state. |

Persist locally at most every 32 completed training updates and at epoch end. Stage immutable files first, verify lengths and SHA-256 or equivalent remote object identity, then advance the selecting manifest with the expected parent revision. Record the pinned revision and cursor. Exactly one writer owns a run. Resume by comparing compatible histories and verifying all selected files before advancing. Preserve any local-ahead state. On throttling, obey server retry guidance. If the unuploaded backlog would exhaust scratch space, pause at a complete update. Never erase the sole surviving copy. A sudden session loss can lose work since the last verified upload. The 30 minutes is the publishing cadence, not a guarantee during transport failures.

## Feasibility and decision rules

At FP32, one student's raw training logits occupy 0.108 GB on CIFAR-10, 1.08 GB on CIFAR-100 or 4.32 GB on Tiny ImageNet. Across 75 student fits these alone total **121.824 GB** decimal. The five live-KD teacher streams add 5.4 GB. Teacher-fitting evidence, all epoch weights, cache bodies, validation rows, CSV, optimizer state and retained remote versions add further storage. Before scheduling the full campaign, measure bytes per family, budget retained remote versions and repository file counts, and compare peak staging plus a 2 GB working reserve to actual free space. Group small files into time-range shards with directly readable CSV and manifest indexes. Use observed seconds per epoch plus cache, audit and upload times to estimate each stage. No Kaggle runtime or HF storage entitlement is assumed. [HF storage guidance](https://huggingface.co/docs/hub/en/storage-limits) · [Rate limits](https://huggingface.co/docs/hub/en/rate-limits)

Report the five paired CIFAR-100 ResNet-18 accuracy differences in percentage points with their mean and two-sided 95% paired t interval. First compare guided against shuffled, then against uniform if the first lower bound exceeds zero. A practical recommendation requires both lower bounds above zero and at least +0.30 points mean guided-minus-shuffled. A gain only against shuffled means the score ranks images usefully while uniform may still be the better allocation. Show the three-seed replications per dataset and model, including any reversals. Secondary outcomes include per-class recall, NLL, calibration and total teacher queries. Energy break-even compares cache creation plus repeated cached fits against repeated live fits at the same quality margin. Live controls support that estimate only for the primary ResNet-18/CIFAR-100 setting. Report costs descriptively elsewhere.

**Status.** The prior primary trainer has local synthetic verification. The Kaggle input adapters, mobile/transfer runner, all-stage timed publication and live cross-session recovery need implementation or qualification for this expanded campaign. No benchmark fit, HF upload of training evidence, accuracy improvement or energy saving is claimed by this plan.

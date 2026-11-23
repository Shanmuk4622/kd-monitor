# Kaggle execution and evidence guide

For the current three Kaggle inputs and fully crossed student comparisons, use the [revised experiment plan](KAGGLE_EXPERIMENT_PLAN.md). The binary-archive examples here describe the previous notebook.

## Run the supplied preflight

1. Import [00_cache_preflight.ipynb](../notebooks/00_cache_preflight.ipynb) into a fresh Kaggle Python notebook. CPU suffices for the first numerical check. Select dual T4 to exercise both target devices. The preflight has embedded sources and uses the installed NumPy/PyTorch stack.
2. To produce the real CIFAR-100 manifest, attach the **binary** archive named cifar-100-binary.tar.gz as read-only input. The expected publisher MD5 is 03b5dce01913d631647c71ecec9e9cb8. If it is attached multiple times, set CIFAR100_ARCHIVE to the intended exact path in the first code cell. The preflight can run offline without any secret. [Official editions](https://www.cs.toronto.edu/~kriz/cifar.html)
3. Choose **Run All**. The notebook makes a new fixture directory under /kaggle/working/kd_view_preflight, prints per-device numerical checks and exercises a synthetic overlap case. With the archive attached it also writes split_manifest.csv and split_receipt.json. Without it the real-data status prints as pending. Previous fixture outputs are preserved.
4. Inspect policy_checks.csv, pilot_scores.csv, the view CSV indexes, dense FP32 NPY bodies and receipt.json. The synthetic fixture has 25 images and four cache policies. A standalone policy consumes 50 pilot and 50 additional image predictions. Verification forwards have their own count. Confirm the SHA-256 associations and the per-device tolerances in [progress](PROGRESS.md). A rerun creates a new disposable receipt.

The preflight produces disposable records. Its numerical source is [cache_preflight.py](../src/cache_preflight.py), its data source is [cifar100_manifest.py](../src/cifar100_manifest.py), and [build_preflight.py](../src/build_preflight.py) regenerates the notebook. The separate [CIFAR-100 master](../notebooks/01_cifar100_master.ipynb) implements training and recovery. Use the [primary experiment instructions](PRIMARY_MASTER.md) for its three modes and the [progress log](PROGRESS.md) for the evidence available so far. The requirements below apply across the study.

## Inputs and immutable identities for the measured master

| Input | Identity to verify and persist |
| --- | --- |
| CIFAR-100 and CIFAR-10 | Use the canonical binary archives. Expected upstream MD5 values are 03b5dce01913d631647c71ecec9e9cb8 and c32a1d4ab5d03f1284b67883e8d87530 respectively. Record actual archive SHA-256, Kaggle input owner/slug/version, class list, split digest, decoded image digests and train-only normalization. |
| Tiny ImageNet | Use tiny-imagenet-200.zip, reference MD5 90528d7ca1a48142e341f4ef8d21d0de. Verify wnids.txt, val_annotations.txt and 200 classes. Keep a PNG/JPEG row's file hash and decoded pixel hash distinct. [Archive record](https://zenodo.org/records/10720917) |
| Teacher and student | Record factory name, small-image stem, package/source hash, initialization seed, class order, normalization, selected teacher epoch and weights SHA-256. Reject a teacher trained on validation candidates or a source with unknown fitting membership. |
| Control block | Bind protocol revision, data, teacher, block seed, init hash, deterministic source schedule, transform version and cache row index to the run identity. Log the actual Kaggle hardware, driver, device UUID, CUDA, PyTorch, torchvision, timm, deterministic settings and start time. |
| Publication | Use one designated KD research dataset repository and versioned namespace. Its exact destination is unset until supplied or created for the measured work. The tyre and earlier KD repositories are independent projects. Only the coordinator reads the Kaggle secret HF_TOKEN. |

The primary delivery is **one sequential Run All notebook** with separate modes for training and the sealed-set analysis. It verifies inputs, exercises recovery, fits a teacher, constructs each bank independently and audits each student. It checks compatible incomplete work and completed stages against recorded identities. The initial CIFAR-100 block precedes the transfer stages.

## Hardware and time boundaries

Place the teacher, cache build and student fits on GPU 0 in all paired scientific arms, with batch 128 on that GPU. Use GPU 1 for an independent replay check before each block. Keep it idle during comparative timing. Log both devices and mark their assigned roles so active-board and both-reserved-board totals can be reconstructed. Randomize the order of arms inside each block with a seeded permutation and record it. Use the same evidence cadence and data-loader settings across arms. Avoid concurrent fits, cache builds or external uploads during measured power windows.

Synchronize CUDA before marking the start and end of each measured phase. Include loading the teacher/targets, replay reads, ranking, training and matched evaluation/writes in declared phase boundaries. Charge actual materialization and serialization to construction. Record idle pre/post readings and a standardized warm-up separately. Record file-system first-read versus reuse and any host-level cache that cannot be controlled. Schedule short, separately timed publications between update windows and charge their observed elapsed/device use to the research ledger.

Check availability and monotonicity of NVML's cumulative energy counter (mJ). If available, use endpoint differences and report joules. Also sample board power with monotonic timestamps at a nominal 200 ms. If the counter is unavailable, integrate watts over seconds by the trapezoidal rule. Record observed cadence, counter resets, stalls and readings adjacent to both boundaries. For integration require at least 98% of phase duration covered, intervals of at most one second and measured boundary readings within 0.25 s. A suspect boundary, large gap or unsupported sensor gets a reason and missing energy. Preserve valid accuracy and timing records. Never convert a missing sensor reading into zero. Report board scope and wall-clock separately. [NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

## Storage and evidence contract

A single four-view FP32 logit body uses 4 × N × classes × 4 bytes. At the planned fitting sizes, this is **7.2 MB for CIFAR-10, 72 MB for CIFAR-100 and 288 MB for Tiny ImageNet** (decimal). Indexes, schedules, audit reads, activation RAM, versions and evidence increase the actual cost. The full pre-update student logits alone require 1.08 GB for one CIFAR-100 student and 4.32 GB for one Tiny ImageNet student at 60 epochs. Budget one active fit at a time.

Measure per-family formatted bytes in a representative dry epoch. Project unique local/remote bytes, retained epoch weights, rolling optimizer state, previous complete state, temporary preparation/downloads and peak staging. Require the measured peak projection plus a 2 GB working reserve to fit the actual free local quota. Treat 20 GB as a planning assumption. If the full run does not fit, stage immutable verified shards, verify the pinned copy before reclaiming reproducible local copies, or revise the budget explicitly before fitting. Preserve unsent evidence. Estimate actual run hours from the same T4 dry epoch, with separate build, validation and transport overheads.

| Evidence family | Grain and keys | Fields, units and trigger |
| --- | --- | --- |
| Provenance/splits | One row per source; source ID | Role, class where permitted, original path/index, source and pixel SHA-256, exclusion reason. Freeze once. |
| Teacher and policies | One row per image/block and one per bank/slot | Two probe scores in nats, zero/one/two correct probes, class rank, quota, slot, crop pixels/flip, teacher identity, FP32 logit row and hash. Write at cache commit. |
| Actual exposure | One row per observation; run/update/draw | Epoch, source ID, augmentation key, view slot or fresh key, label, pre-update argmax, true-class probability, NPY row/offset for full logits, sample weight. Increment on a complete update. |
| Optimization | One row per optimizer update | Examples, CE and KL sums, temperature/weight, total objective, LR, gradient norm, parameter version, duration and any retry. Keep sums and denominators. |
| Validation and test | One row per image/checkpoint/role | Label, top class, logits or lossless native logit reference, correctness, uncertainty bin. Record each validation epoch and the frozen final releases. Persist per-class counts, confusion and paired predictions. |
| Compute and energy | One row per device observation and phase | Monotonic seconds, UUID/role, watts or mJ, used/peak memory, synchronized boundaries, missingness, actual calls, wall time, build/read bytes. |
| Recovery and audit | One row per commit/publish/failure | Update cursor, immutable shard ranges/hashes, complete state, owner, source/protocol compatibility, remote revision/receipt, verified scope, retry/backoff and local-ahead status. |

Use direct UTF-8 CSV for tables, native NPY/PT for dense arrays and weights, JSON for configuration and selecting manifests. Stable integer draw offsets and model versions must link native rows to CSV. Keep integer counts, int64 timing/cursors where appropriate, FP32 logits, FP64 accumulation for scores and energy, and explicit missingness reasons. Retain raw sources, state pointers, epoch weights, errors, run receipts and derived tables through paper audit. Rolling complete optimizer state plus epoch weights covers restart and endpoint review. Historical per-parameter gradients, activations and every past optimizer slot are explicit omissions. Exercise the promised writers, inspect record types/ranges and produce per-family row-count/byte coverage before measured training.

## Checkpoints, publication and repair

Commit state only at a completed optimizer update. Flush immutable evidence shards, record the committed CSV/native byte or row prefixes, atomically replace the complete state and then advance a selecting pointer. Save model parameters/buffers, optimizer momentum, LR schedule/update, Python/NumPy/CPU/CUDA RNG, epoch and next draw, partial sums, retries, view-bank/schedule hashes and source/schema/environment identity. FP32 has no scaler. The logit recorded by a training forward refers to the pre-update weights.

One coordinator owns a fenced lease. Workers cannot advance the same remote pointer. Check identity and the current remote parent before publishing. Batch publication at approximately 30 minutes, each major stage, and a catchable stop/error. Respect rate-limit headers and bounded retries with jitter. Preserve local-ahead work and the last verified remote revision. Store directly browsable CSV alongside native bodies. [Hub rate-limit documentation](https://huggingface.co/docs/hub/en/rate-limits)

On a catchable stop, stop sampling new updates, finish or roll back the active transaction, flush and attempt one consistent publication subject to server backoff. On restart, compare local and remote durable state, verify hashes and committed prefixes, discard only uncommitted tails and continue at the next cursor. A hard kill with lost scratch can recover only from the newest verified compatible remote revision. Keep the original traceback, partial artifacts and receipt if any gate fails.

## Release gates and independent review

| Before | Evidence required |
| --- | --- |
| A measured teacher or student fit | Pinned inputs and code, true model/batch shapes, train-only role audit, counted teacher calls, live-versus-cache agreement, actual dual-T4 telemetry, writer coverage, measured storage and time forecast. |
| A long measured run | Interrupt after a complete update and mid-write, then resume in a fresh process. Compare the next model, optimizer, scheduler, RNG, data cursor and CSV/native prefixes to an uninterrupted trajectory. Inject corrupt/tail/lease conflicts and verify refusal or rollback. Demonstrate one real publish/download/fresh-session continuation in the dedicated KD namespace. |
| A paper result | Retain the executed master, selected and final checkpoint hashes, pinned revisions, per-seed predictions, failure inventory and exact inclusion counts. Independently recompute top-1, paired deltas, intervals, calibration, query bills and board-energy totals from the direct records. Label existence-only checks and fully downloaded/recalculated checks separately. |

Record execution status, scientific outcome and promotion as distinct fields. If guided allocation fails, retain the original decision rule and the negative cases. The design contains no deployment promotion.

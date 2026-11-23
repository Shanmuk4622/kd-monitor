# Run the four-worker KD experiment on Kaggle

The private results repository is [Shanmuk4622/kd-crop-budget-4w-2026](https://huggingface.co/datasets/Shanmuk4622/kd-crop-budget-4w-2026). All six notebooks come with the repository, GROUP = "all" and NUM_WORKERS = 4 already set. Each training notebook also has its own WORKER_ID. Use all files from the same release.

## Set up each Kaggle notebook

Import the corresponding .ipynb into Kaggle. Choose **GPU T4 x2**, turn **Internet** on, attach these three datasets using **Add Input**, and grant the notebook access to a Kaggle secret named **HF_TOKEN** with write permission for the private repository. Select **Run All**. The first code cell needs no edits. Mount discovery records file hashes and split membership. If Kaggle exposes a numeric version in the mount path, it is also recorded. If automatic discovery reports ambiguous mounts, set INPUT_PATHS to the exact folders shown by Kaggle.

| Dataset | Exact input |
| --- | --- |
| CIFAR-10 | [shanmuk4622/dataset-cifar10-python](https://www.kaggle.com/datasets/shanmuk4622/dataset-cifar10-python) |
| CIFAR-100 | [shanmuk4622/dataset-cifar100-python](https://www.kaggle.com/datasets/shanmuk4622/dataset-cifar100-python) |
| Tiny ImageNet | [shanmuk4622/tinyimagenet200](https://www.kaggle.com/datasets/shanmuk4622/tinyimagenet200) |

### Tiny ImageNet input for NB01

Attach version 1 of the Tiny ImageNet input above. The six notebooks use this owner and slug by default. The expected folder is `/kaggle/input/datasets/shanmuk4622/tinyimagenet200/tiny-imagenet-200/`. It contains `train/`, `val/`, `test/`, `wnids.txt` and `words.txt`. The reader uses 100,000 training images and 10,000 official validation images. The latter form the final labeled holdout. The official test folder has no role in scoring.

To switch from a running notebook, cancel the old run, start a fresh session with the updated NB01 and attach the three inputs above. The path printed for Tiny ImageNet should contain `shanmuk4622/tinyimagenet200`. Keep the same HF results repository. The new release can reuse the two completed CIFAR capacity qualifications when the source release, input identity, model checks and software/hardware environment match. Look for `Reused verified CIFAR qualification` on the first run and `Verified previous qualification` on later runs.

NB01 prints the directory it is listing, the number of images decoded and hashed, and the start of each capacity probe. During a session, a status line prints the current stage roughly every minute. Each Tiny ImageNet probe runs one fitting epoch (704 batch updates). The displayed committed-update count advances at local checkpoint boundaries. Wait for `Qualification complete` before launching the four workers.

## Run order

| When | Notebook | Responsibility |
| --- | --- | --- |
| First, once | 01_Kaggle_Preflight.ipynb | Qualify the three inputs, run nine model/dataset capacity probes and publish the recovery anchor. Wait for “Qualification complete.” Review its disk and timing forecasts. |
| Parallel, worker 0 | 02_Kaggle_Train.ipynb | CIFAR-100 teacher, then assigned student blocks. |
| Parallel, worker 1 | 02_Kaggle_Train_Worker1.ipynb | CIFAR-10 teacher, then assigned student blocks. |
| Parallel, worker 2 | 02_Kaggle_Train_Worker2.ipynb | Tiny ImageNet teacher, then assigned student blocks. |
| Parallel, worker 3 | 02_Kaggle_Train_Worker3.ipynb | Independent hard-label controls, then assigned student blocks. |
| Last, once | 03_Kaggle_Results.ipynb | Verify all 75 student audits, evaluate the two frozen cohorts and publish paper tables. Wait for students_scored to reach 75. |

Start each worker in a separate Kaggle session. Four simultaneous sessions require eight T4s across your available quota. If capacity is limited, prioritize publishing all three teachers. A worker waiting for a teacher can be interrupted normally and resumed with the same ID after the owner finishes its teacher. No worker depends on another worker's student fits. The 78 planned fits consist of three teachers and 75 students. Paired students reuse 38 unique teacher prediction banks across the campaign.

### Running on a newer Kaggle image

All six Kaggle notebooks use the software recorded by the completed preflight, including Python 3.12.13 and PyTorch 2.10.0+cu128. Select **GPU T4 x2**, enable Internet and choose **Run All**. If the current kernel matches, the stage runs directly. Otherwise the notebook installs the required Python and GPU packages under `/kaggle/temp`. Allow several GB of download and disk space on the first run in each fresh session.

For this campaign, NB01 has already completed. Leave active workers running. Import the updated notebook for each worker when starting or resuming its session. Each training notebook uses its own worker ID and monitoring secret and prints `Verified preflight source, environment and input for` after checking each dataset receipt. Use NB03 after all student fits and audits finish.

If the environment check passes and a dataset identity check fails, verify the attached dataset edition and mount against NB01.

## Choosing work for a Kaggle session

Leave an active notebook running. At its next restart import the refreshed file with the same worker ID. All 78 assignments and the original preflight identity are preserved. NB01 is already complete for this campaign.

| First-cell setting | Example | Effect |
| --- | --- | --- |
| `MAX_RUNS` | `2` | Select up to two unfinished fits assigned to this worker, including required owned teachers. Zero selects all. |
| `MODEL_FILTER` | `["resnet18"]` | Select assigned fits of these architectures. An empty list allows all models. |
| `SESSION_HOURS` | `8.5` | Use a soft wall-clock budget from the start of planning. Setup before planning and the final upload are additional. Zero disables the limit. |

Startup checks the published fit and audit manifests, skips verified completions, and prints the selected run IDs, epochs and update counts. The planner prioritizes ready jobs and resumable work while maintaining teacher and prediction-bank ownership. The first dataset loads while the independent GPU recovery check runs. On a clean Kaggle image, package installation and verifying the dataset still take time. A time budget stops training at a checkpoint; give the final upload time to finish before closing Kaggle.

## Evidence in Hugging Face

| Granularity | Files and measurements |
| --- | --- |
| Every epoch | `epochs/NNN/metrics.csv` stores fit count, cross-entropy sum, distillation KL sum where applicable, objective sum, training correct count, validation size, validation correct count, and mean validation NLL. Compute training or validation accuracy as correct divided by count. Epochs also retain weights, validation logits and per-image labels and predictions. |
| Every optimizer update | `updates/NNNNNNNN/updates.csv` includes batch and epoch, sample count, CE/KL/objective sums, correct count, learning rate, gradient L2 norm, seconds and peak allocated GPU bytes. Each file holds the updates since the previous local checkpoint. Exposure rows and logits link predictions to specific input views. |
| Per model and run | Configuration, optimizer and RNG recovery state, best/final weight references, audit, timing windows, and board energy when available. Final held-out evaluation is performed by NB03. |

Local evidence is checkpointed during training. HF publication uses grouped commits around the 30-minute window and at boundaries. Up to 128 changed evidence files share a staging commit; a verified manifest selects the checkpoint. An abrupt disconnect can lose local updates since the last selected HF commit. Supabase heartbeats do not publish training evidence to HF.

## Live worker monitoring

Follow the [Supabase setup guide](../monitoring/SETUP.md) to connect the four training notebooks to the live dashboard. It lists the SQL queries, Kaggle secrets and website configuration.

## Training evidence and recovery

Open the repository's `kd-view-budget/four-workers-tinyimagenet200-v1/` directory for this release. Prior CIFAR receipts remain in `kd-view-budget/four-workers-v1/`. Each run or bank has a MANIFEST.json selecting hashed evidence. Training retains per-update and per-epoch records, logits, validation outputs, weights, optimizer and RNG state, teacher-query counts, timing and available board telemetry. The publisher targets 30-minute commits during long-running work and commits at major boundaries. A status heartbeat on its own is not a recoverable training checkpoint. Last verified commits remain the recovery points during network outages. Raw student training logits alone are projected at 121.824 GB across the full campaign. Allow additional space for weights, evidence, bank files and retained revisions.

To resume, reopen the **same worker** with the same three inputs and secret, then select **Run All**. Never run two copies with the same ID concurrently. A hard-killed worker's lease can take up to 45 minutes after its last renewal to expire. A teacher wait names the owning worker. A mismatched input hash requires restoring the qualified input. If storage qualification fails, inspect the reported required bytes and scratch reserve. Per-session transport logs and local receipts are under /kaggle/working/kd_v2_receipts/<role>/.

## Paper results

The results notebook saves the following files to the paper-results unit and /kaggle/working/kd_v2_receipts/results/. Each scored run also contains per-image predictions, logits and confusion matrices.

| Final file | Contents |
| --- | --- |
| `accuracy.csv` | 150 model/cohort rows with denominators, correct counts, top-1, NLL, calibration and macro recall. |
| `paired.json` | Guided-minus-random and guided-minus-uniform effects by dataset and architecture, individual seed differences and intervals. |
| `resources.csv` | Measured active-board joules, both-board joules, wall time and sensor window counts for fits and caches. |
| `reuse.csv` | CIFAR-100/ResNet-18 live-versus-cached energy at one, three and five student uses, and the quality-margin check. Blank energy indicates unavailable measurements. |


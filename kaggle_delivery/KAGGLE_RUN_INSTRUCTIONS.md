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


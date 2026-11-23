# Running the CIFAR-100 experiment

**Input edition.** This document describes the earlier binary-reader notebook. For the requested Python-edition Kaggle inputs and the expanded campaign, use the [current experiment plan](KAGGLE_EXPERIMENT_PLAN.md).

23 November 2026. The [training notebook](../notebooks/01_cifar100_master.ipynb) contains its own Python sources and the frozen [protocol](PROTOCOL.md). Import the notebook into a fresh kernel. The first code cell contains all four user settings. Each mode runs from top to bottom with **Run All**.

| Mode | What it does | Inputs |
| --- | --- | --- |
| `qualification` (default) | Fits a small CNN on disposable images, checks all six configurations, reconstructs their losses and exercises a deliberate crash and fresh-process restart. | Local Python environment with the notebook's dependencies or a Kaggle notebook. No data archive or publishing credentials are needed. |
| `measured` | Fits one ResNet-34 and each selected seed's six ResNet-18 students. Builds the four target banks independently and audits the records for each fit. | Original CIFAR-100 binary archive, Kaggle with two T4 GPUs, Internet enabled, and write access to a dedicated Hugging Face dataset. |
| `final_review` | Retrieves the five blocks' 30 final checkpoints and checks that their audits identify the same completed fits. Scores the frozen final cohort and writes the paired analysis. | Same official input, target environment and repository as measured training. The full set of primary checkpoints and reviews must already exist. |

## Measured training

1. Attach the original `cifar-100-binary.tar.gz` as a read-only Kaggle input. The published MD5 is `03b5dce01913d631647c71ecec9e9cb8`. The runner computes its SHA-256, freezes source IDs, audits decoded-pixel overlap and derives normalization from the fitting rows. Set `CIFAR100_ARCHIVE` only if automatic detection finds multiple or no matching files. Record the Kaggle dataset owner, slug and version in the session's run notes. [Official CIFAR distribution](https://www.cs.toronto.edu/~kriz/cifar.html)
2. Select **GPU T4 ×2**, enable Internet, and grant the notebook access to a Kaggle secret named `HF_TOKEN` with write permission to the intended dedicated Hugging Face **dataset** repository. Set `HF_DATASET_REPO` to its existing `owner/repository` identifier. The repository name is the user's project destination. Keep the token in the secret store.
3. Set `MODE = "measured"` and leave `SEED_BLOCKS = (42,)` for the first block. The notebook first runs disposable GPU checks, publishes a disposable checkpoint, downloads it at its exact revision and continues training in a fresh Python process in the same Kaggle session. It then projects peak storage and time from one disposable epoch per model. Those checks establish the target session's actual capabilities.
4. The teacher uses 90 epochs on the fitting split and is selected by validation accuracy, with validation loss and then earlier epoch breaking ties. Each student uses 60 epochs, batch size 128 and its block's paired initialization and source order. GPU 0 does the timed scientific work. GPU 1 verifies replay independently outside those windows. Follow the printed run IDs and archived fit reviews.
5. To finish the primary comparison, use `SEED_BLOCKS = (42, 123, 2026, 31415, 27182)` with the same archive, source, environment and repository. A compatible completed teacher or student is reused. A single seed gives a pilot estimate only. After all 30 student fits, select `MODE = "final_review"` to produce the analysis. The final endpoint is student epoch 60.

## How the six configurations answer the question

| Student configuration | Teacher calls during the 60-epoch fit or its standalone preparation | Purpose |
| --- | ---: | --- |
| Hard labels, fresh crops | 0 | Establish the supervised student's performance. |
| Live KD, fresh crops | 2,700,000 | Establish what the student learns when the teacher sees every training view. |
| Four cached views per image | 180,000 | Test the simple equal allocation. |
| Random six/two within each class | 180,000 | Test the same unequal quotas with random recipients. |
| Entropy-ranked six/two | 180,000 | Test whether ordinary uncertainty identifies the useful images. |
| Disagreement-ranked six/two | 180,000 | Test whether a change in the teacher's predictions between two crops identifies the useful images. |

Each cached policy evaluates two initial views of every image and spends its remaining 90,000 evaluations according to the policy. A high disagreement score means the teacher's distribution across classes changed across the two views. The most probable class can stay the same. A separate check compares the original scores with another two views on ten fitting images per class. Its 2,000 evaluations per guided block go to a diagnostic ledger. Across methods, every student still receives one exposure to every fitting image each epoch.

## What to inspect

| Output | Use |
| --- | --- |
| Split manifest and receipt | Inspect stable source IDs, exclusions, fitting and eligible validation counts. Final-label columns are empty at setup. |
| Bank `scores.csv`, `views.csv` and `logits.npy` | Join image ID and view slot to the raw teacher distribution, rank score and quota. |
| Each `updates/<ending-update>/` directory | Join `exposure.csv` and `updates.csv` to `student_logits.npy`. Live KD also writes aligned `live_teacher_logits.npy`. Cached KD references the corresponding bank row. |
| `epochs/<epoch>/`, `ENDPOINT.json` and fit review | Inspect validation predictions and checkpoint hashes. Verify training objectives, exact source coverage, validation accuracy and selection. |
| `measurement/`, `setup_telemetry/` and `transport.jsonl` | Inspect raw device readings, accepted integration method, elapsed time, warm-up and idle windows, revision IDs and errors. The two-board total includes the reserved verification GPU. |
| Final `results.csv`, `paired_result.json`, `resources.csv` and `reuse_scenarios.csv` | Inspect denominators on both the deduplicated and full test cohorts, paired gains in percentage points, board energy and reuse projections. Per-run prediction rows, raw logits and class confusion matrices accompany the tables. |

One actual CPU fixture shard is included as an example. Row `j` of each array corresponds to `logit_row = j` in the exposure table. Each update row records its sum and denominator. [Exposure](../evidence/primary_master/live_example/exposure.csv) · [Updates](../evidence/primary_master/live_example/updates.csv) · [Student logits](../evidence/primary_master/live_example/student_logits.npy) · [Live teacher logits](../evidence/primary_master/live_example/live_teacher_logits.npy)

## Interruption and interpretation

Local recovery selects a complete optimizer update and its matching evidence. Routine writes group up to 32 updates, with an additional commit at each epoch boundary or catchable stop. The coordinator attempts publication at roughly 30-minute intervals, at stage boundaries and on a handled stop. A hard loss of Kaggle working storage can discard everything after the latest **verified remote** revision. Preserve the executed notebook and its transport log. To resume, use the same settings in a fresh session. An active competing writer or incompatible identity stops the run and preserves the existing evidence.

The capacity forecast distinguishes optimizer compute, elapsed local training including evidence writes and the separately observed cost of transport. A live independent-session restore, target quotas and real board sensors still require verification. Keep a missing energy measurement blank. Cache reuse projections at one, three and five students use the measured build and per-fit costs. The separate greenfield totals charge the trained teacher to each alternative. These numbers refer to training-board energy on the recorded hardware.

The primary comparison estimates the value of choosing the recipients of the extra crops for one fixed teacher. Prefer the guided rule only if it clears the [prespecified comparisons](PROTOCOL.md). A gain over shuffled assignment alongside a loss to uniform assignment would show that the score identifies relatively useful images while uniform allocation still trains the stronger student. The MobileNetV4, CIFAR-10 and Tiny ImageNet transfer stages have their own entries in the [run matrix](RUN_MATRIX.csv). The notebook here implements the CIFAR-100 residual-network stage.

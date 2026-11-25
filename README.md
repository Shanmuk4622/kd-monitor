# Where should a small CNN spend its teacher queries?

## Current Kaggle release

Start with [KAGGLE_RUN_INSTRUCTIONS.md](notebooks/KAGGLE_RUN_INSTRUCTIONS.md). Run preflight once, the four workers pulling ready fits from the shared Supabase queue in separate sessions, then results once. The revised matrix contains 78 fits.

Knowledge distillation study · design version 1.0 · 22 November 2026

A teacher can give a student more than the correct label. Two crops of a picture may both be labeled “fox,” while the teacher's predictions move from fox to dog when the face is cropped out. It costs a teacher evaluation to learn what happened on each crop. This study asks whether those changes tell us which training images deserve extra evaluations.

Each of the four caching methods buys **180,000 teacher predictions** for 45,000 CIFAR-100 fitting images. One stores predictions for four crops per image. Another stores six for half the images in each class and two for the rest, chosen at random. The proposed method gives the six crops to images for which two initial teacher predictions disagree most. Those initial predictions count toward the budget. Every student sees every fitting image once per epoch. Beating random allocation would show that the choice of images matters. Beating uniform allocation would establish that the choice improves on a simpler way to use the same budget.

The [live worker monitoring guide](monitoring/SETUP.md) includes the Supabase schema, payloads, Kaggle secret setup and complete static dashboard.

## Start here

All six campaign notebooks use [shanmuk4622/tinyimagenet200](https://www.kaggle.com/datasets/shanmuk4622/tinyimagenet200) for Tiny ImageNet.

The six Kaggle notebooks are in `notebooks/`. Start with [01_Kaggle_Preflight.ipynb](notebooks/01_Kaggle_Preflight.ipynb), launch the four training notebooks listed in the [run instructions](notebooks/KAGGLE_RUN_INSTRUCTIONS.md), then run [03_Kaggle_Results.ipynb](notebooks/03_Kaggle_Results.ipynb).

**Current campaign.** The [Kaggle experiment plan](research/KAGGLE_EXPERIMENT_PLAN.md) and [78-fit run matrix](research/RUN_MATRIX_V2.csv) cover the three specified Kaggle datasets, two student architectures and HF checkpoint schedule. The binary-reader notebook below documents the earlier implementation. The revised plan specifies the input adapters needed for measured training.

| File | What it answers |
| --- | --- |
| [Project context](research/CONTEXT.md) | What claim are we testing, what decisions are fixed, and where can work resume? |
| [Literature review](research/LITERATURE.md) | What do the two supplied papers establish, and which other approaches are closest? |
| [Filled experiment brief](research/EXPERIMENT_BRIEF.md) | What is the question, the primary comparison and the decision rule? |
| [Experimental protocol](research/PROTOCOL.md) | What exactly are the datasets, CNNs, configurations, seed blocks and cost boundaries? |
| [Run matrix](research/RUN_MATRIX.csv) | Which teacher or student runs in each block, which cache is reused, and what does each fit cost in calls and updates? |
| [Kaggle and evidence guide](research/RUN_GUIDE.md) | What runs today, how will the measured stage recover, and which records must exist? |
| [Paper draft](research/PAPER_DRAFT.md) | How could the question, rationale and methods be presented in a paper? |
| [Progress and verification](research/PROGRESS.md) | Which files and numerical checks exist, and which results are still pending? |
| [Decisions, mistakes and improvements](research/DECISIONS_AND_CORRECTIONS.md) | Which assumptions changed, why, and what would require a new protocol version? |
| [First Kaggle notebook](notebooks/00_cache_preflight.ipynb) | Run the disposable cache/replay checks and optionally audit the original CIFAR-100 archive. |
| [CIFAR-100 training notebook](notebooks/01_cifar100_master.ipynb) | Run qualification, the primary ResNet teacher and six student configurations, or the final analysis. |
| [Primary experiment instructions](research/PRIMARY_MASTER.md) | Which notebook settings, inputs and credentials does each stage need? Where will I find the predictions and recovery records? |

The comparison starts with a ResNet-34 teacher and a ResNet-18 student on CIFAR-100. Further experiments change the student to MobileNetV4-Conv-Small on the same data, or keep the residual pair and change the dataset to CIFAR-10 or Tiny ImageNet. This layout lets us interpret an architecture change separately from a dataset change. The maximum budget is 57 student fits and three dataset-specific teachers. The first measured block is one teacher and six students.

**Implementation update, 23 November 2026.** Both notebooks have executed locally. On synthetic data, the second notebook completed all six student configurations on CPU and a GTX 1650. A fresh process reproduced the uninterrupted model, optimizer state, random-number state and all 72 training predictions after a deliberate crash. Official CIFAR-100 training, on-target Kaggle telemetry and a live checkpoint transfer are pending. The mobile and additional-dataset experiments are specified in the run matrix.

## Original project inputs

The two PDFs remain at the project root. Their titles, file digests and relevant pages are in the literature review. The supplied [experiment handoff](https://chatgpt.com/library?file_id=libfile_5ea50c3a6da481918874897eba59e758&account_id=personal) and [blank experiment brief](https://chatgpt.com/library?file_id=libfile_1448a3ce80fc8191bcb84dc9d61e7e2b&account_id=personal) provided the record-keeping and execution requirements. This project uses its own image data, model identities, run records and publication destination.

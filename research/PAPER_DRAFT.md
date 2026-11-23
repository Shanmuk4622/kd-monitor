# Do Crop-Sensitive Images Deserve More Teacher Queries? A Fixed-Budget Study of CNN Distillation

Working manuscript · introduction and methods before data collection · 22 November 2026

## Abstract

Precomputing a teacher's predictions lets multiple small networks learn from the same supervision. When the teacher can evaluate only a few augmentations of each training image, a practical question arises. Should every image receive the same number of evaluations? We design a controlled comparison in which a frozen teacher sees two views of every fitting image and assigns the remaining evaluations according to how much its predictions change. A class-balanced random control has the same distribution of view counts. A uniform control uses the same total number of evaluations. The student sees every fitting image once per epoch in all three cases. Our primary comparison uses a ResNet-34 teacher and a ResNet-18 student on CIFAR-100, with a mobile CNN and two additional benchmarks specified for replication. We define how to measure both the accuracy gained from choosing the images and the GPU energy required to build and reuse the resulting supervision. This draft specifies the study before its first measured fit.

## Introduction

A small image classifier can learn distinctions that a single class label cannot express. For an image labeled “fox,” a teacher might assign nearly all its probability to fox when the face is visible, yet spread substantial probability over fox and dog when the crop contains only fur. If the student sees both crops, the two probability vectors provide different supervision. A class label alone assigns the same target to both. Temperature-scaled distillation provides the mechanism for transferring this information. [Hinton, Vinyals and Dean](https://arxiv.org/abs/1503.02531)

The resources required to obtain that supervision depend on where the measurement begins. Rafat and colleagues counted the cost of searching distillation temperatures and showed how batch-wise sampling could avoid much of that search. Rocco di Torrepadula and colleagues started from an inference-energy constraint on an edge platform and chose a student and distillation settings around it. Their findings motivate precise cost accounting. Neither reported saving gives a laboratory the cost of evaluating its particular teacher on another crop. [Rafat et al.](https://doi.org/10.1371/journal.pone.0285668) · [Rocco di Torrepadula et al.](https://doi.org/10.1016/j.sysarc.2026.103692)

Prior work already stores predictions for augmented views, aligns teacher and student inputs and reuses stored supervision across architectures. Confidence-based crop handling and teacher–student divergence have also informed which examples to emphasize. These results leave a narrower empirical question. At a fixed number of teacher calls, does variation in the teacher's predictions across two crops of the same image reveal where extra cached views are worth buying? A prediction change can reflect useful visual ambiguity. It can also mean the crop removed the object. Either mechanism could dominate. [FKD](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4359_ECCV_2022_paper.php) · [Consistent teaching](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html) · [Dataset reinforcement](https://machinelearning.apple.com/research/reinforce-data) · [FerKD](https://arxiv.org/abs/2312.17473) · [Adaptive sampling](https://openaccess.thecvf.com/content/ICCV2025/papers/Chae_What_to_Distill_Fast_Knowledge_Distillation_with_Adaptive_Sampling_ICCV_2025_paper.pdf)

## Methods

We allocate four teacher queries per fitting image on average. Two initial queries are common to all policies. For the proposed policy, we calculate the Jensen–Shannon divergence between their class distributions and rank images within each true class. The upper half receives four more views, bringing its total to six. The lower half retains its two initial targets. A shuffled policy assigns the six-view slots randomly within the same class. A uniform policy stores four aligned targets for every image. The number of target vectors, their precision and the number of student updates agree across these policies. The differing assignment is observable in both the saved index and the actual pre-update exposure records.

The primary endpoint is the paired change in held-out CIFAR-100 top-1 accuracy between disagreement-based and shuffled allocation. The comparison with uniform caching determines whether varying the quotas offers a practical advantage. Mean teacher entropy supplies a secondary ranking rule, and live KD and conventional supervised training locate the finite-bank results on a familiar accuracy scale. Each primary block begins from the same student initialization across arms and independently redraws candidate views and the random allocation. We retain every block's prediction counts. One teacher is fixed for each dataset, which limits inference about other trained teachers.

The energy analysis separates bank creation from use. We count the initial probes, ranking, serialization, reads and subsequent fits. A cache earns back its build cost after the cumulative per-fit savings exceed that cost. We report both active-board energy and the energy of both reserved boards when supported, alongside wall-clock times and sensor coverage. The final student has the same inference architecture across the allocation arms. [Measurement interface](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

## Evidence required for the results section

| Intended result | Required evidence | Interpretation |
| --- | --- | --- |
| Primary paired comparison | Five paired final-epoch differences, raw correct counts, audited denominator and interval | Establish whether the image ranking earns extra accuracy at fixed cost. |
| Uniform and entropy controls | Same teacher, splits, block schedule and actual view ledger | Distinguish the value of changing quotas from the value of this particular score. |
| Cost and reuse | Direct phase timing, valid board readings, target counts and bytes | Identify any supported accuracy-preserving energy saving and its reuse count. |
| Transfer and failure cases | Mobile CNN at fixed dataset, two dataset changes at fixed student, class supports and independent-view diagnostics | Describe where the allocation succeeds, is inconclusive, or harms transfer. |

## Discussion boundaries

If the proposed method beats the shuffled control but trails uniform caching, the ranking carries information while equal allocation remains the better practical choice in that setting. A win over both still needs an explanation for the extra value of disagreement over simple entropy. If all finite banks lose materially to live KD, the diversity of fresh views becomes a plausible constraint on this recipe. If the teacher is unstable on background crops, allocating more of those crops may reinforce an unreliable signal. The recorded controls distinguish these outcomes, although separating the effects of crop diversity and soft labels would require a matched finite-bank hard-label follow-up.

The empirical abstract, results, conclusions and any recommendation await measured evidence. Use the full [protocol](PROTOCOL.md) and the [evidence register](LITERATURE.md) when extending this draft.

# Literature review and evidence register

Reviewed 22 November 2026. The two supplied PDFs motivate the energy question. The closest comparisons for the proposed allocation rule come from work on cached supervision, consistent augmentations and adaptive data selection.

## The two supplied papers

| Reference | What was measured | What it establishes for this study |
| --- | --- | --- |
| Rocco di Torrepadula, Maisto, Cilardo and Mazzocca (2026), *Distilling knowledge for low-energy AIoT*, Journal of Systems Architecture 174, 103692. Supplied PDF 0004, §§3–5 and Tables 1–2. | The authors choose a student depth under a target inference-energy budget, then guide the KD temperature and imitation weight using the teacher, capacity gap and dataset. They evaluate related ResNet and DenseNet variants. Section 4.2 describes 20 ms rail sampling on the ZCU102 during deployed inference. Tiny ImageNet is a ResNet experiment. | Their reported reduction of up to about 80% is hardware- and comparison-specific. We can use the discipline of defining the energy budget while measuring a different quantity, the GPU energy spent to create and consume teacher targets. [Publisher](https://doi.org/10.1016/j.sysarc.2026.103692) |
| Rafat et al. (2023), *Mitigating carbon footprint for knowledge distillation based deep learning model compression*, PLOS ONE 18(5), e0285668. Supplied PDF 0005, pp. 8–10 and Tables 1, 4, 7. | A ResNet-18 teacher supervises a MobileNetV2 student in the image-classification comparison. Sampling temperature by batch avoids a search across temperatures. On CIFAR-10 the authors report 91.78% for tuned KD and 91.67% for stochastic KD, with 3.99 and 0.21 kWh respectively. | The roughly 19-fold difference includes the search avoided. It cannot be used to predict the cost of one live teacher pass or the savings from caching. Our ledger charges preparation, each fit, and any failed or exploratory runs separately. [Open paper](https://doi.org/10.1371/journal.pone.0285668) |

Section 5.3 of the AIoT paper gives a separate training-time example for SubResNet-14 on CIFAR-10. The authors estimate about 87.5 hours for 45 configurations and about 17.8 hours for their reduced search including brief probes. We preserve that distinction between a search campaign, a single student fit and energy per deployed inference. [Rocco di Torrepadula et al.](https://doi.org/10.1016/j.sysarc.2026.103692)

The local inputs have the following SHA-256 digests.

| Local PDF | Pages | File digest |
| --- | ---: | --- |
| DOC-20260726-WA0004..pdf | 14 | 84b2b8279b2eee99228de6017d9243cf862089b40ab5348dc1e00c7e66beec7b |
| DOC-20260726-WA0005..pdf | 22 | a0f09f8a6fb5bb1036b08ca14fafb56b76216b33dd119d50ee342bc94778b854 |

## Closest prior work

| Work | What it already does | Consequence for the experiment |
| --- | --- | --- |
| Hinton, Vinyals and Dean (2015), *Distilling the Knowledge in a Neural Network* | Uses softened teacher outputs to supervise a smaller model. | Keep conventional logit distillation as the common objective and change only how cached views are assigned. [Paper](https://arxiv.org/abs/1503.02531) |
| Beyer et al. (CVPR 2022), *Knowledge Distillation: A Good Teacher Is Patient and Consistent* | Examines how matching the teacher's and student's input views affects distillation. | Reconstruct the exact crop and flip for each saved teacher output. Test logits, loss and student gradients against a live teacher on the same pixels. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html) |
| Shen and Xing (ECCV 2022), *A Fast Knowledge Distillation Framework for Visual Recognition* | FKD precomputes multi-crop targets and replays them during training. | Uniform four-view caching is the closest simple baseline. It uses FKD's principle under our crop budget. An exact reproduction of their ImageNet recipe would be a different experiment. [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4359_ECCV_2022_paper.php) |
| Faghri et al. (ICCV 2023), *Reinforce Data, Multiply Impact* | Stores augmentation metadata and teacher predictions for use by multiple architectures, including a reinforced CIFAR-100 dataset. | Cache reuse and spreading its cost across students already have precedent. Measure the actual construction and read costs, and state when amortization is only projected. [Paper and code](https://machinelearning.apple.com/research/reinforce-data) |
| Shen (ICCV 2023), *FerKD* | Mines and calibrates region-level supervision using the reliability of teacher labels. | An unstable crop can contain background or remove the object. Retain reliability diagnostics and include an entropy-ranked control. [Paper](https://arxiv.org/abs/2312.17473) |
| Chae et al. (ICCV 2025), *What to Distill? Fast Knowledge Distillation with Adaptive Sampling* | KDAS uses teacher–student divergence to update the sampled training set and calibrates the loss. | Adaptive sample selection is established. Here the teacher alone allocates a reusable bank before student initialization. Every original fitting image still has equal exposure. [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Chae_What_to_Distill_Fast_Knowledge_Distillation_with_Adaptive_Sampling_ICCV_2025_paper.pdf) |
| Frank et al. (CVPR 2025), *What Makes a Good Dataset for Knowledge Distillation?* | Studies the importance of class coverage and input diversity for transfer with substitute distillation data. | Rank within ground-truth classes. Check class-level recall and exposure so a few classes cannot absorb the entire budget. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Frank_What_Makes_a_Good_Dataset_for_Knowledge_Distillation_CVPR_2025_paper.pdf) |

## Candidate contribution and its limits

The falsifiable contribution is a controlled test of **which source images should receive extra cached augmentations when teacher calls, dense-logit bytes, per-class quotas and student updates are fixed**. The key comparison permutes who receives the extra views. The uniform baseline answers whether a practitioner benefits from varying the count at all. A confidence-based ranking tests whether disagreement adds anything beyond ordinary teacher uncertainty.

Existing papers already motivate every individual ingredient. The proposed paper stands on the quality of the controlled comparison, its negative cases, and measurement of the cost of obtaining supervision. A priority claim about adaptive KD, caching, or the use of disagreement would exceed this review. The search covered publisher/CVF/ECVA pages and arXiv full text for fast KD, reinforced datasets, adaptive KD sampling, view disagreement, cache budgets and teacher-query cost. It is a targeted review, with a final overlap check due before submission.

## Data, model and measurement sources

| Resource | Purpose |
| --- | --- |
| Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* and CIFAR release | Original counts, fine-class labels, file format and archive checksums. [Dataset and report](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Le and Yang, Tiny ImageNet | Native 64 × 64 images, 200 classes and original challenge partitions. An archived copy has a published checksum. [Stanford report](https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf) · [Archive](https://zenodo.org/records/10720917) |
| He et al., *Deep Residual Learning for Image Recognition*, and Qin et al., *MobileNetV4* | Residual primary pair and mobile-family replication. The small-image stems are study-specific. [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) · [MobileNetV4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5647_ECCV_2024_paper.php) · [Implementation identity](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) |
| NVIDIA Management Library | Device energy counters, board power units and feature support. A hosted GPU may expose only part of this interface. [Device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html) |

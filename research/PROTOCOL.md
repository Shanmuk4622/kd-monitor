# Experimental protocol

Version 1.0 · 22 November 2026 · specified before benchmark training. Freeze a digest of this document, the implementation and the input manifests before the first measured student update. Later changes receive a dated amendment.

## 1. Primary comparison

For one fixed ResNet-34 teacher, estimate the change in final ResNet-18 accuracy when extra teacher views go to source images with the largest two-view Jensen–Shannon divergence. The comparator randomly assigns exactly the same six/two quotas within each fine class. The estimand includes the resulting differences in the training views. A claim that soft labels alone caused a gain would require additional hard-label runs on the identical finite view banks.

Illustration on CIFAR-100: each fine class contributes 450 fitting images. Uniform allocation buys 450 × 4 = 1,800 teacher evaluations per class. Unequal allocation buys 225 × 6 + 225 × 2 = 1,800. Across 100 fine classes, the total is 180,000. One query means one augmented image forwarded through the teacher, irrespective of batching.

## 2. Datasets and splits

| Dataset | Native input | Fitting images | Validation candidates | Final cohort before exact-overlap audit |
| --- | --- | ---: | ---: | --- |
| CIFAR-100 | 32 × 32 RGB, 100 fine classes | 45,000, 450/class | 5,000 | 10,000 official test images |
| CIFAR-10 | 32 × 32 RGB, 10 classes | 45,000, 4,500/class | 5,000 | 10,000 official test images |
| Tiny ImageNet-200 | 64 × 64 RGB, 200 synsets | 90,000, 450/class | 10,000 from official training | 10,000 labeled official validation images, reserved as the study's final holdout |

Within each fine class, sort official training IDs by SHA-256 of the UTF-8 string `kd-view-v1|1729|<image_id>`, tie-breaking on the canonical ID. Use the first fitting quota and reserve the remainder for validation. CIFAR-100 binary IDs are `train/00000` through `train/49999` in source order. CIFAR-10 uses official batch order followed by record order and the same five-digit ID convention. Tiny ImageNet uses the synset and original relative POSIX path, with class order fixed by wnids.txt. Use val_annotations.txt for its final labels. The challenge test directory has no role.

Hash decoded RGB uint8 in CHW order and record the native dimensions. Keep fitting exposure fixed. Exclude validation candidates matching fitting pixels and repeated validation pixels after the first canonical source ID. Exclude final images matching any official training candidate and repeated final pixels after the first. Freeze the resulting IDs without reading final predictions. Keep and report exclusion reasons, per-class eligible counts and standard unfiltered final-split results. Exact hashing does not identify every near-duplicate.

Compute channel means and population standard deviations on unaugmented fitting pixels alone. For validation and final evaluation, use native resolution, RGB conversion and this normalization. Reserve final labels and metrics for one release after all predeclared reconfirmation blocks are fixed. Validation can be used for teacher selection and debugging. Version changes after seeing benchmark outcomes must disclose the development history.

## 3. Networks and common training recipe

| Role | Configuration | Purpose |
| --- | --- | --- |
| Teacher | torchvision ResNet-34, one per dataset, trained from scratch, seed 907 | One strong, fully identified target for all policies on that dataset |
| Primary student | torchvision ResNet-18, trained from scratch | Same-family capacity reduction |
| Cross-family student | timm MobileNetV4-Conv-Small with pretrained weights disabled | Tests whether allocation transfers to a different CNN on the same CIFAR-100 split |

For the two ResNets, use a 3 × 3 first convolution with 64 outputs, stride 1, padding 1, no bias, and an identity in place of the initial max pool. Keep all subsequent residual stages and global pooling. For MobileNetV4 keep the native convolutional stages and adaptive pooling, set the first convolution stride to 1, use the dataset class count, and set dropout and stochastic depth to zero. Record the exact variant, package version, input/output shapes, parameter counts and effective stage strides at qualification. The native 224-pixel weights and published device timings are not part of this comparison.

Train in FP32 with SGD, momentum 0.9, no Nesterov, weight decay 0.0005 on trainable tensors, global batch 128 on one active GPU. Include the final short batch. Use a peak learning rate of 0.1, a five-epoch linear warm-up and cosine decay to zero at the last update. For total updates U, warm-up W and zero-based update u, use 0.1(u+1)/W for u < W and 0.05[1+cos(π(u−W)/(U−W−1))] after warm-up. No gradient accumulation, mixed precision, label smoothing, MixUp, CutMix or per-policy tuning. Each seed shares identical starting weights and minibatch order across its arms. Disable nondeterministic kernels and record the actual software/hardware identity.

Teachers use fresh crops, hard-label cross-entropy and 90 epochs. Pick the highest validation top-1, breaking ties by lower validation NLL and then earlier epoch. Record the validation margin over the supervised student once available. Retain weak or anomalous teachers in the audit. Students use 60 epochs, so every two-, four- and six-view bank can be traversed equally often. The epoch-60 checkpoint defines the main comparison. Save the validation-best checkpoint separately under its own identity. A short training horizon constrains the interpretation.

## 4. Query allocation and loss

1. Reflect-pad CIFAR by four pixels or Tiny ImageNet by eight. Enumerate all native-size integer crops and both horizontal flip states. Order the possible transforms per source image with NumPy PCG64 seeded from a stable hash of the block seed and source ID. Slots 0–5 are distinct transform keys. Every policy in a block shares their definition. Repeated pixels can still occur in uniform regions.
2. Evaluate slots 0 and 1 with the frozen teacher in evaluation mode. Store dense raw FP32 logits. Calculate the two softmax vectors at scoring temperature 1. Let their mean be m. The disagreement score is half of KL(p₀‖m) + KL(p₁‖m), using natural log. The entropy control uses the mean of H(p₀) and H(p₁). Log teacher correctness on both fitting probes as a diagnostic.
3. Rank within each ground-truth class. In a ranked arm, assign six views to the top half and two to the bottom half. In the shuffled arm, rank by a seeded hash independent of teacher outputs. In an odd-sized class the median gets four. Break score ties with an independent seeded hash and canonical ID. Evaluate only the extra slots assigned. Include both initial probes in each policy's budget. Never evaluate six views everywhere and then charge only the retained four.
4. Replay the saved crop and flip that produced the target. Each fitting image appears once per epoch, in a block-specific shuffle shared across arms. Select slot (epoch + per-image offset) modulo its assigned count. Across 60 epochs a two-slot source has 30 exposures per slot, a four-slot source 15, and a six-slot source 10. Batching the flat cache rows uniformly would overweight six-slot sources and change the question.
5. For the fresh arms, use the same per-epoch source order and draw one crop/flip independently from the same transform family using a separate keyed stream. The hard-label and live-KD arms share these draws. For KD, minimize (1−α)CE(y,zₛ) + αT² KL(softmax(zₜ/T) ‖ softmax(zₛ/T)), with α=0.5, T=4 and a batch-mean KL. The student and teacher see identical pixels. Cached logits are raw and can reproduce this loss. These values are fixed design choices.

| Arm | Target production | Student exposure | Question answered |
| --- | --- | --- | --- |
| Hard labels | No teacher | Fresh views | How strong is supervised training? |
| Live KD | Frozen teacher sees every fresh student view | Fresh views | What difference accompanies replacing the fresh view stream with a finite bank? |
| Uniform four | Four cached aligned targets per image | Finite bank | Does choosing quotas beat a simple reusable cache? |
| Shuffled six/two | Class-balanced, randomly assigned quotas | Finite bank | Does the identity of the images given extra views matter? |
| Entropy six/two | Highest mean teacher entropy gets six | Finite bank | Is ordinary uncertainty sufficient? |
| Disagreement six/two | Highest two-view disagreement gets six | Finite bank | Proposed allocation |

Live KD and uniform caching differ in view diversity and target delivery. A no-fit replay test comparing both paths on exactly the same tensors handles the numerical equivalence question separately. The entropy arm and any supervised-versus-live comparison are secondary.

## 5. Run sequence and realistic scale

| Order | Dataset and student | Arms | Paired blocks | Student fits |
| ---: | --- | --- | --- | ---: |
| 1 | CIFAR-100 / ResNet-18 | All six above | 42, 123, 2026, 31415, 27182 | 30 |
| 2 | CIFAR-100 / MobileNetV4-Conv-Small | Uniform, shuffled, disagreement | 42, 123, 2026 | 9 |
| 3 | CIFAR-10 / ResNet-18 | Uniform, shuffled, disagreement | 42, 123, 2026 | 9 |
| 4 | Tiny ImageNet / ResNet-18 | Uniform, shuffled, disagreement | 42, 123, 2026 | 9 |

One teacher per dataset adds three fits to the maximum of 57 students. Begin with the CIFAR-100 teacher and seed-42 arms. Count the seed-42 fits once if the protocol survives qualification. Qualify code on disposable fixtures first. Keep benchmark test labels sealed during the first block. Funding, time and engineering integrity can determine whether the later blocks execute. The sign of the pilot gain is not a reason to discard a seed or silently change the confirmatory set.

The [machine-readable run matrix](RUN_MATRIX.csv) fixes the order, cache ownership and planned query counts. Within each block it sorts arms by the SHA-256 of `order-v1|dataset|model|seed|arm` using the CSV spellings. CIFAR-100 cross-family runs reuse the corresponding same-seed banks. The complete matrix contains 38 distinct banks and 8,460,000 planned physical construction queries. The five full live-KD fits add 13,500,000 teacher training-view queries. These are budget calculations. The separate diagnostic forwards, parity checks, teacher validation, retries and actual energy must enter the observed ledger.

At batch 128, CIFAR has 352 updates per epoch and Tiny ImageNet 704. Each student therefore needs 21,120 or 42,240 updates, respectively. The maximum design totals 1,393,920 student updates and 126,720 teacher updates. Book time using the median seconds per update from a representative T4 dry run, plus separately timed cache construction, evaluation, recovery and publication. Do not substitute the local fixture speed. If resources permit only the first block, report it as a one-seed pilot.

## 6. Statistical decisions and interpretation

For block s, compute Δₛ = 100(correct_guided − correct_shuffled)/N on the identical final IDs. Always show the five values, raw correct counts and N. Report their mean, sample SD and two-sided 95% interval using t with four degrees of freedom. The interval describes paired training-block variation conditional on this teacher and finite evaluation cohort. Its interpretation is approximate with five blocks. Images, epochs and crops are not extra independent training replicates. Report per-image paired disagreements as a diagnostic.

First test disagreement versus shuffled. Only if its lower bound exceeds zero, test disagreement versus uniform with the same interval in fixed order. A useful allocation recommendation requires both lower bounds above zero and a primary point estimate at least +0.30 percentage points. This design threshold corresponds to about 30 decisions per 10,000 images. If an upper bound falls below +0.30, that comparison excludes a gain of the specified practical magnitude. An interval spanning both zero and +0.30 is inconclusive. Report estimates and intervals even when formal testing stops. Show entropy, class and transfer comparisons descriptively. Multiple datasets sharing raw source families do not constitute independent populations.

Report macro recall, per-class support/confusion, NLL, and top-label ECE with 15 equal-width confidence bins on [0,1], including confidence 1 in the last bin. Inspect high-score fitting images with correct and incorrect teacher probes. For a diagnostic independent of the ranking views, sample ten fitting IDs per primary CIFAR-100 class by a separate hash. On those 1,000 images measure disagreement on two new transform keys outside the six-slot bank. Report the within-class rank association and charge the 2,000 additional queries to diagnostics. This check cannot influence allocations or epochs.

## 7. Actual energy and projected reuse

For each standalone strategy log physical teacher queries, unique cache rows, dense payload bytes, CSV/index and total disk bytes, wall time, GPU memory peaks and GPU-board joules for each phase. For the clean comparison, build each bank independently (including the first two queries for each image) and reuse it only for a declared second student on the same dataset/teacher/seed/view contract. Reuse between student architectures on CIFAR-100 is feasible by design. Charge the physical creation once in the actual research ledger. Show the full standalone preparation cost for each strategy's scenario. Include fixtures, failures and exploratory costs in the campaign ledger under separate phases.

For a fixed teacher and one reusable bank, let B be the measured active-board cache-build energy including ranking and serialization. Let L and C be mean measured live and cached per-fit energies, including reads, the same evaluation cadence and matched local evidence writing. A reused bank saves energy after r fits when B + rC < rL. If L > C, the first strict saving is max(1, floor(B/(L−C))+1). If L ≤ C and B ≥ 0 there is no finite saving. Give the observed single-fit total and show scenarios r = 1, 3, 5. For unequal students use the sum of their paired per-fit differences. Label extrapolations, report per-block values, and flag denominators close to zero.

Describe an energy saving as quality-preserving only when the lower paired 95% bound for cached minus live top-1 exceeds −0.50 percentage points on the primary pair. Reuse this tolerance only for that secondary interpretation. Charge teacher training explicitly for a greenfield deployment scenario and for comparison with hard labels. The same teacher cost cancels in a like-for-like comparison of two KD alternatives. Neither the teacher nor its cache is required for student inference. An accuracy difference between identically shaped students cannot establish an inference-energy saving.

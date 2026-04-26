# Vehicle Re-Identification — Knowledge Base

A consolidated reference for the models, dimensions, losses, and hyper-parameters used in this coursework, alongside the published methods that motivated them. Tuned for **VeRi-776**.

---

## 1. Project at a glance

- **Task.** Vehicle Re-Identification (vehicle Re-ID): given a query image, retrieve images of the *same* vehicle captured by a *different* camera.
- **Primary dataset.** VeRi-776 (`-s veri -t veri`).
  - 776 vehicle IDs, ~50k images, 20 cameras (urban surveillance).
  - Standard split: 576 train IDs (~37k images), 200 test IDs, 1,678 queries, 11,579 gallery images.
  - Evaluation removes same-camera matches (`--query-remove True`).
- **Metrics.** mAP, CMC Rank-1/5/10/20 (computed in [src/eval_metrics.py](coursework/src/eval_metrics.py)).
- **Training pipeline.** [main.py](coursework/main.py) — bf16 autocast, identity classification (xent) + hard-batch triplet loss, optional auxiliary loss (SupCon for v3), MultiStep LR, DataParallel.

---

## 2. Models implemented (in [coursework/src/models/](coursework/src/models/))

### 2.1 ResNet baselines — [resnet.py](coursework/src/models/resnet.py)

Variants registered: `resnet18`, `resnet18_fc512`, `resnet34`, `resnet34_fc512`, `resnet50`, `resnet50_fc512`.

| Variant | Backbone | last_stride | Pool | FC bottleneck | Embedding dim |
|---|---|---|---|---|---|
| `resnet50` | ResNet-50 (Bottleneck) | 2 | AdaptiveAvgPool2d(1) | — | 2048 |
| `resnet50_fc512` | ResNet-50 | **1** (preserves spatial res) | AdaptiveAvgPool2d(1) | Linear→512 + BN1d + ReLU | **512** |
| `resnet18_fc512` | ResNet-18 (BasicBlock) | 1 | AvgPool | Linear→512 | 512 |
| `resnet34_fc512` | ResNet-34 | 1 | AvgPool | Linear→512 | 512 |

- ImageNet-pretrained weights via `model_zoo.load_url`.
- `last_stride=1` doubles the feature-map size at stage-4 (a "trick" from BoT) — improves Re-ID without extra params.
- Output during training is `(logits, v)` so the trainer can apply both xent and triplet on the *same* embedding `v`.

### 2.2 Torchvision wrappers — [tvmodels.py](coursework/src/models/tvmodels.py)

- `mobilenet_v3_small`: lightweight CNN, ~2.5M params, `feature_dim = backbone.classifier[0].in_features` (576).
- `swin_t_fc512` (torchvision variant — superseded by the timm variant in [`__init__.py`](coursework/src/models/__init__.py)).

### 2.3 Swin-Tiny via timm — [timm_model.py](coursework/src/models/timm_model.py)

| Variant | Backbone | Pool | FC reduce | BNNeck | Embedding dim |
|---|---|---|---|---|---|
| `swin_t_custom` | `swin_tiny_patch4_window7_224` | timm global pool | — | BatchNorm1d(768), bias frozen | **768** |
| `swin_t_fc512` | `swin_tiny_patch4_window7_224` | timm global pool | Linear(768→512) + Kaiming | BatchNorm1d(512), bias frozen | **512** |

- `swin_tiny_patch4_window7_224` requires **224×224** input and uses 7×7 windows; ~28M params.
- Both variants follow **BoT-style BNNeck**: BN before the classifier; `bn.bias.requires_grad_(False)` so the BN bias does not act as a per-class shift.
- Classifier is `Linear(emb_dim, num_classes, bias=False)` with `std=0.001` init (Luo et al. recommendation).
- Training output: `(logits, pre_BN_feat)` so triplet acts on the un-normalised feature, classifier on the BN-normalised one — the BoT/AGW decoupling.

### 2.4 CLIP-SENet variants v1/v2 — [clip_senet.py](coursework/src/models/clip_senet.py)

Both inspired by **CLIP-SENet (arXiv 2502.16815, 2025)** — using a CLIP-pretrained semantic encoder fused with appearance features.

#### `clip_senet_v1_dual` — Dual-stream, both trainable

```
Input → ResNet-50 (timm)   ─→ f_app ∈ ℝ^2048
      → ViT-Tiny/16 (timm) ─→ f_sem ∈ ℝ^192
                              concat → ℝ^2240
                              Linear(2240→512) + BNNeck + Linear(512, num_classes)
```

- 2240 = 2048 (ResNet-50) + 192 (ViT-Tiny patch16/224, ImageNet-21k → ImageNet-1k pretrained).
- *No* explicit AFEM — the linear `reduce` does the fusion.

#### `clip_senet_v2_frozen` — Single-stream, backbone frozen

```
Input → CLIP ViT-B/32 (frozen, vit_base_patch32_clip_224)
      → CLS token ∈ ℝ^768
      → AFEM: Linear(768→48) → ReLU → Linear(48→768) → Sigmoid → multiply
      → Linear(768→512) + BNNeck + Linear(512, num_classes)
```

- AFEM = **Adaptive Fine-grained Enhancement Module**, a Squeeze-and-Excitation gate on the CLS token (reduction ratio 16:1, 768 → 48 → 768).
- Backbone forced to **224×224** via `F.interpolate` because CLIP ViT-B/32 was trained at that resolution.
- Only AFEM + reduce + BN + classifier are trainable — small parameter footprint, no fine-tuning of CLIP.

### 2.5 CLIP-SENet v3 — [clip_senet_v3.py](coursework/src/models/clip_senet_v3.py)

`clip_senet_v3_ibn_supcon` — the most evolved variant.

```
Input → ResNet-50-IBN-a (torchreid, ImageNet+IBN pretrained) → f_app ∈ ℝ^2048
      → ViT-Tiny/16 (timm, ImageNet pretrained)              → f_sem ∈ ℝ^192
                                              AFEM(192) on f_sem  ↓
                                                              residual SE: x*σ(W2 ReLU(W1 x))+x
                                                              concat 2048 + 192 = 2240
                                                              Linear(2240→512) + BNNeck + Linear(512, num_classes)

Aux loss → SupConLoss on the 512-d feature, weight 0.5, T=0.07
```

| Block | Detail |
|---|---|
| Appearance backbone | `torchreid.models.build_model("resnet50_ibn_a")` — IBN-Net mixes Instance Norm + Batch Norm in early layers, robust to camera-style shift. Output 2048-d after avg-pool. |
| Semantic backbone | `timm.create_model("vit_tiny_patch16_224")` — 192-d CLS feature. |
| AFEM(192) | `Linear(192→24) → ReLU → Linear(24→192) → Sigmoid`, **with residual** (`x*attn(x) + x`) — differs from v2 (which has no residual). Reduction 8:1. |
| Fusion | concat(2048, 192) = 2240 → Linear → 512. |
| Aux loss | **SupConLoss** (Khosla et al., NeurIPS 2020): batch-wise contrastive loss using identity labels as positives. Temperature 0.07, weight 0.5. |
| Hook | `model.aux_loss(features, pids)` is called by [main.py](coursework/main.py) (lines 412-413) and added to `lambda_xent*xent + lambda_htri*htri`. |

---

## 3. Comparison summary (this repo)

| Model | Backbone(s) | Input HxW | Pretraining | Embedding | Params (approx.) | Training output | Aux signal |
|---|---|---|---|---|---|---|---|
| `resnet50_fc512` | ResNet-50 | 256×128 default* | ImageNet-1k | 512 | ~25M | logits, v | — |
| `mobilenet_v3_small` | MBv3-small | 256×128 | ImageNet-1k | 576 | ~2.5M | logits, v | — |
| `swin_t_fc512` (timm) | Swin-T | **224×224** | ImageNet-1k | 512 | ~28M + heads | logits, reduced_feat | — |
| `swin_t_custom` (timm) | Swin-T | 224×224 | ImageNet-1k | 768 | ~28M + heads | logits, global_feat | — |
| `clip_senet_v1_dual` | ResNet-50 + ViT-Tiny | 224×224 | ImageNet-1k | 512 (from 2240) | ~30M | logits, feat | — |
| `clip_senet_v2_frozen` | CLIP ViT-B/32 (frozen) | 224×224 (forced) | CLIP (LAION-style) | 512 (from 768) | ~88M backbone frozen, ~0.5M trainable | logits, feat | AFEM gate |
| `clip_senet_v3_ibn_supcon` | ResNet-50-IBN-a + ViT-Tiny | 224×224 | ImageNet+IBN, ImageNet | 512 (from 2240) | ~30M | logits, feat | **SupCon** + AFEM |

\* args defaults are `--height 128 --width 256` but [train.sh](coursework/train.sh) overrides with `--height 224 --width 224` for Swin/CLIP compatibility.

### 3.1 Embedding-dimension rationale

- **2048-d ResNet-50** pooled output is the standard CNN feature size.
- **192-d ViT-Tiny** is a small, semantic prior — TinyCLIP-style.
- **768-d CLIP ViT-B/32** is the standard CLIP image encoder dim.
- **512-d** is the chosen retrieval dim across all variants — keeps memory/runtime constant for fair comparison and matches `resnet50_fc512` from Li et al. (Array 2022).
- **BNNeck** before the classifier (Luo et al., BoT 2019): classification operates on `BN(feat)`, retrieval/triplet on the **un-normalised** `feat` — decouples the two loss geometries.

---

## 4. Losses & training mechanics

| Component | Source | Default | Notes |
|---|---|---|---|
| CrossEntropy | [src/losses](coursework/src/losses) `CrossEntropyLoss(label_smooth)` | `--label-smooth` flag | Label smoothing ε ≈ 0.1 (typical) |
| Triplet | `TripletLoss(margin=0.3)` | `--margin 0.3` | Hard-batch triplet (Hermans et al. 2017) |
| SupCon (v3 only) | [clip_senet_v3.py](coursework/src/models/clip_senet_v3.py) | T=0.07, weight=0.5 | Per-batch contrastive loss using PIDs |
| Loss combination | `loss = λ_xent·xent + λ_htri·htri (+ aux)` | λ_xent=1, λ_htri=1 | Aux added if `model.aux_loss` exists |
| DeepSupervision | wraps multi-output heads | — | Applied when `outputs`/`features` is tuple |
| Mixed precision | `torch.autocast(bfloat16)` | enabled if CUDA | Train and test |

---

## 5. Hyper-parameters — defaults vs. VeRi-776 recipe

Defaults from [args.py](coursework/args.py); the active overrides are in [train.sh](coursework/train.sh).

| Hyper-parameter | argparse default | VeRi-776 recipe ([train.sh](coursework/train.sh)) | Comment |
|---|---|---|---|
| `--source-names / --target-names` | required | `veri / veri` | Single-domain Re-ID |
| `--height × --width` | 128 × 256 | **224 × 224** | Swin/CLIP need 224² |
| `--train-sampler` | `RandomSampler` | inherited | For triplet, `RandomIdentitySampler` is preferred (P×K) — current sampler is generic random |
| `--num-instances (K)` | 4 | inherited | Used only with identity sampler |
| `--optim` | `adam` | `amsgrad` | AMSGrad variant of Adam |
| `--lr` | 3e-4 | 3e-4 | Same |
| `--weight-decay` | 5e-4 | inherited | Standard Re-ID value |
| `--adam-beta1 / beta2` | 0.9 / 0.999 | inherited | — |
| `--momentum` | 0.9 | unused | SGD only |
| `--lr-scheduler` | `multi_step` | inherited | |
| `--stepsize` | `[20, 40]` | `[20, 40]` | Decay points |
| `--gamma` | 0.1 | inherited | LR×0.1 at each step |
| `--max-epoch` | 60 | **10** in current recipe | shortened for AMP/Colab runs |
| `--train-batch-size` | 32 | **64** | |
| `--test-batch-size` | 100 | 100 | |
| `--margin` | 0.3 | 0.3 | Triplet margin |
| `--lambda-xent / lambda-htri` | 1 / 1 | 1 / 1 | |
| `--label-smooth` | off | not enabled | Toggle on for BoT parity |
| `--random-erase / color-jitter / blur-aug / crop-aug / color-aug` | off | off | Augmentations live in [src/data_manager.py](coursework/src/data_manager.py) |
| `--seed` | 1 | inherited | |
| `--query-remove` | True | True | Drops same-camera matches |

### 5.1 BoT/AGW recipe parity check

| Trick (BoT 2019 / AGW 2021) | Implemented? |
|---|---|
| Warmup LR | ❌ — only MultiStep is wired up. AGW recommends linear warmup over the first 10 epochs |
| Random Erasing | ⚠ flag exists (`--random-erase`) but not enabled in `train.sh` |
| Label Smoothing | ⚠ flag exists, not enabled |
| Last-stride = 1 | ✅ in `*_fc512` variants |
| BNNeck | ✅ in Swin/CLIP variants; classic ResNet variants use plain FC head |
| Generalised-Mean (GeM) pooling (AGW) | ❌ AdaptiveAvgPool used instead |
| Weighted Regularised Triplet (AGW) | ❌ vanilla `TripletLoss(0.3)` used |
| Non-local attention (AGW) | ❌ |
| `RandomIdentitySampler` (P×K) | ❌ — `RandomSampler` configured; switching to identity sampler unlocks proper hard-batch mining |

Enabling `--random-erase`, `--label-smooth`, and switching to `RandomIdentitySampler` are the cheapest wins toward the published BoT/AGW numbers.

---

## 6. Published methods — context for the comparison

Numbers below are **reported in their respective papers** — not produced by this repo. They are the targets the local models are benchmarked against.

### Person Re-ID (used as conceptual templates)

| Method | Year | Backbone | Market-1501 mAP / R1 | DukeMTMC mAP / R1 | Key idea |
|---|---|---|---|---|---|
| **BoT** (CVPRW '19) | 2019 | ResNet-50 | 85.9 / 94.5 | 76.4 / 86.4 | BNNeck + warmup + last-stride=1 + RE |
| **AGW** (TPAMI '21) | 2021 | ResNet-50 + non-local + GeM | 88.2 / — | 79.6 / — (mINP 45.7) | WRT loss + GeM + non-local; introduces mINP |
| **TransReID** (ICCV '21) | 2021 | ViT-B/16 (DeiT) | 89.5 / 95.2 | — | Pure-Transformer + JPM + SIE |

### Vehicle Re-ID (direct competitors)

| Method | Year | Backbone | VeRi-776 mAP / R1 | VehicleID R1 (small) | Key idea |
|---|---|---|---|---|---|
| `resnet50_fc512` (Li et al., Array 2022) | 2016/22 | ResNet-50 | baseline (lower than Swin-T) | baseline | CNN baseline |
| `swin_t_fc512` (Li et al., Array 2022) | 2021/22 | Swin-T | improved over ResNet baseline | improved | Hierarchical Transformer |
| **VAMI** (CVPR '18) | 2018 | CNN + GAN | SOTA at publication | SOTA at publication | Viewpoint-aware GAN-inferred features |
| **TransReID** (ICCV '21) | 2021 | DeiT-B/16 (with stride change) | **82.3 / 97.1** | 85.2 | Pure Transformer + JPM + SIE |
| **GLSIPNet** (Array '22) | 2022 | CNN, part-based | improved over PCB | competitive | Global–local similarity-weighted parts |
| **CLIP-SENet** (arXiv '25) | 2025 | CNN + TinyCLIP + AFEM | **92.9 / 98.7** | 90.4 (R5 98.7) | CLIP image encoder + AFEM, no text branch |

VeRi-Wild adds: CLIP-SENet 89.1 mAP / 97.9 Rank-1.

### What our local models inherit from each

| Local model | Drawn from |
|---|---|
| `resnet50_fc512`, `resnet18/34_fc512` | He et al. (ResNet, CVPR '16) — last-stride=1 trick from BoT, `fc512` bottleneck following Li et al. (Array '22) |
| `swin_t_fc512`, `swin_t_custom` | Liu et al. (Swin, ICCV '21) + Li et al. (Array '22) + BNNeck from BoT |
| `clip_senet_v1_dual` | Dual-stream interpretation of CLIP-SENet — appearance (CNN) + semantic (small ViT) |
| `clip_senet_v2_frozen` | CLIP-SENet's "use a frozen pretrained CLIP" angle, with explicit AFEM gate |
| `clip_senet_v3_ibn_supcon` | CLIP-SENet + IBN-Net robustness (Pan et al., ECCV '18) + SupCon (Khosla et al., NeurIPS '20) |

---

## 7. Quick reference — VeRi-776 launch command

```bash
STUDENT_ID=ss50456 STUDENT_NAME="Sourav Sen" python main.py \
  -s veri -t veri \
  -a clip_senet_v3_ibn_supcon \
  --root /content \
  --height 224 --width 224 \
  --optim amsgrad --lr 0.0003 \
  --max-epoch 60 --stepsize 20 40 --gamma 0.1 \
  --train-batch-size 64 --test-batch-size 100 \
  --margin 0.3 --lambda-xent 1 --lambda-htri 1 \
  --label-smooth --random-erase \
  --eval-freq 10 \
  --save-dir logs/clip_senet_v3-veri
```

Tips for closing the gap to published CLIP-SENet numbers:
1. Enable `--label-smooth` and `--random-erase`.
2. Switch the sampler to `RandomIdentitySampler` (P=16, K=4) so triplet mining sees enough positives per ID.
3. Add a 10-epoch linear LR warmup before the MultiStep decay.
4. Increase `--max-epoch` to 60–120 if compute allows; 10 epochs (current `train.sh`) is a smoke-test budget.
5. Keep input at 224² for any Swin/CLIP backbone; revert to 256×128 only for pure CNN baselines.

---

## 8. PaperSummary — VeRi-776 numbers from the [papers/](coursework/papers/) folder

This section consolidates **only what the downloaded papers say** about each of the four local models. Where a paper does not report on VeRi-776, that gap is noted explicitly.

### 8.1 `resnet50_fc512`

| Field | Value |
|---|---|
| Paper (architecture) | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — [1512.03385_ResNet_He2016.pdf](coursework/papers/1512.03385_ResNet_He2016.pdf) |
| VeRi-776 numbers in this paper | **None** — original ResNet paper benchmarks ImageNet/CIFAR/COCO only |
| Best VeRi-776 mAP / Rank-1 (from other papers in folder) | **76.4 / 95.2** — TransReID Table 2 (ResNet-50 backbone, 256×256 input), [2102.04378_TransReID_He2021.pdf](coursework/papers/2102.04378_TransReID_He2021.pdf) p. 6 |
| Other ResNet variants on VeRi-776 (TransReID Table 2, p. 6) | ResNet-101 → 76.9 / 95.2 · ResNet-152 → 77.1 / 95.9 · ResNeSt-50 → 77.6 / 96.2 · ResNeSt-200 → 77.9 / 96.4 |
| ResNet-50-IBN on VeRi-776 (CLIP-SENet Table VI, p. 7 of [2502.16815_CLIP-SENet_2025.pdf](coursework/papers/2502.16815_CLIP-SENet_2025.pdf)) | **91.1 mAP / 97.4 R1 / 98.6 R5** (with full SEM + AFEM head, not pure ResNet-50) |
| Recipe (BoT, p. 2 §2 + p. 3-6 §3, [1903.07071_BoT_Luo2019.pdf](coursework/papers/1903.07071_BoT_Luo2019.pdf)) | Adam lr **3.5e-4** with **MultiStep ×0.1 at epoch 40, 70**, **120 epochs** total · **Warmup**: linear 3.5e-5 → 3.5e-4 over first 10 epochs (§3.1, p. 3) · **last_stride=1** (§3.4, p. 4) · **BNNeck** before classifier, classifier with no bias (§3.5, p. 4-5) · **Random Erasing** p=0.5, S∈(0.02, 0.4), r∈(0.3, 3.33) (§3.2, p. 4) · **Label smoothing** ε=0.1 (§3.3, p. 4) · **Triplet margin** α=0.3 (§3.6, p. 5) · **Center loss** weight β=5e-4 (§3.6, p. 6) · **P=16, K=4**, batch=64 (§2 step 2, p. 2) · Image **256×128** for person; image-size ablation (Table 6, p. 7) shows 224×224 / 384×128 / 384×192 all within ±0.5 mAP |
| TransReID's ResNet-50 baseline recipe for VeRi-776 (§4.2, p. 5-6) | Image **256×256** (vehicles), aug = h-flip + pad + random crop + random erasing, batch 64 (4 imgs/ID), **SGD** mom=0.9, wd=1e-4, **lr=0.008** with cosine decay, FP16, single V100 |

### 8.2 `clip_senet` (v1, v2, v3)

| Field | Value |
|---|---|
| Paper | Lu et al., *CLIP-SENet: CLIP-based Semantic Enhancement Network for Vehicle Re-identification*, IEEE T-ITS / arXiv:2502.16815, 2025 — [2502.16815_CLIP-SENet_2025.pdf](coursework/papers/2502.16815_CLIP-SENet_2025.pdf) |
| Best VeRi-776 (Table I, p. 6) | **CLIP-SENet\* (with camera + viewpoint): 92.9 mAP / 98.7 R1 / 99.1 R5** |
| Baseline (ResNeXt101-IBN only, no SEM/AFEM) | **86.7 mAP / 96.8 R1 / 97.9 R5** (Table IV, p. 7, §IV.D.1) |
| Component ablation on VeRi-776 (Table IV, p. 7) | + SEM only → 87.3 / 97.0 · + SEM + AFEM → 91.4 / 97.6 · + SEM + AFEM + camera/viewpoint → 92.9 / 98.7 |
| Backbone ablation on VeRi-776 (Table VI, p. 7) | ResNet-50-IBN → 91.1 / 97.4 / 98.6 · ResNet-101-IBN → 91.9 / 98.2 / 98.9 · SE-ResNet-101-IBN → 91.0 / 97.8 / 98.7 · **ResNeXt-101-IBN → 92.9 / 98.7 / 99.1** (best) |
| AFEM groups ablation (Table V, p. 7) | G=4 → 90.2 / 97.2 · G=8 → 91.1 / 97.6 · G=16 → 91.7 / 98.2 · **G=32 → 92.9 / 98.7 (best)** · G=64 → 90.3 / 97.9 · G=128 → 90.6 / 97.9 |
| Special hyperparameters (§IV.B.1, p. 6-7) | **Image 320×320**, ADAM optimizer, **cosine annealing**, **initial lr 5e-4**, **batch 128 = 16 IDs × 8 imgs/ID** (P=16, K=8 identity sampler), **24 epochs** on VeRi-776 (vs 120 with WarmupMultiStepLR for VehicleID/VeRi-Wild at lr 3.5e-4), random seed **3407**, 1×NVIDIA A40 |
| Loss (§III.C, Eq. 6 p. 5 + Eq. 7 p. 5 + Eq. 8 p. 5) | **Smooth CE** (ε=0.1) + **SupCon**: L = L\_CE + L\_SupCon (1:1, no extra weight). Temperature τ in SupCon — paper uses common 0.07 (matches our `clip_senet_v3.py`) |
| AFEM design difference vs. ours | Paper uses **fully connected attribute grouping with G=32** (§III.C, Eq. 4, p. 5) — input split into G+1 vectors, G grouped + 1 residual, learnable per-group weights w\_i. Our `clip_senet_v3.py` uses a simpler **squeeze-excite (192→24→192) + residual** instead of the grouped-FC variant |
| Re-ranking | **Applied as post-processing only on VeRi-776** (§IV.B.2, p. 7) — counted toward the 92.9/98.7 figure |
| Loss-function ablation (Fig. 3, p. 7) | SupCon outperforms triplet on both baseline and CLIP-SENet at every epoch checkpoint (8/12/16/20/24) |

### 8.3 `swin_t` (`swin_t_fc512`, `swin_t_custom`)

| Field | Value |
|---|---|
| Direct paper (Swin-T applied to VeRi-776) | Li et al., *Vehicle Re-identification method based on Swin-Transformer network*, **Array 16 (2022) 100255** — [1-s2.0-S2590005622000881-main.pdf](coursework/papers/1-s2.0-S2590005622000881-main.pdf). This is the canonical citation our `swin_t_fc512` is named after. |
| Best VeRi-776 mAP / Rank-1 (Li et al. Table 2, p. 6) | **Swin Transformer → 78.6 mAP / 97.3 Rank-1** (best in their table) — *narrowly* beats TransReID 78.2 / 96.5 on Rank-1 (+0.8) and slightly on mAP (+0.4) |
| Other VeRi-776 baselines they report (Table 2, p. 6) | VAMI 50.1 / 77.0 · PROVID 53.4 / 81.6 · VSTP 58.8 / 86.4 · SSL 61.1 / 88.6 · RAM 61.5 / 88.6 · QD-DLF 61.8 / 88.5 · MTCRO 62.6 / 88.0 · TransReID 78.2 / 96.5 |
| Special hyperparameters (§3, p. 5) | **Input 256×256**, **window size 4×4**, **batch 32**, **60 epochs** (loss stabilises ≈ epoch 60, Figs. 4-5 p. 6 + Fig. 7 p. 7); augmentations = random flip + random erase + random crop; ImageNet-pretrained Swin defaults; 1× RTX 3060 (12 GB) |
| Caveats with Li et al. 2022 | Paper is short and omits: (a) explicit Swin variant — *Tiny vs Small vs Base* not stated; community follow-ups assume **Swin-Tiny**; (b) optimiser, learning rate, and scheduler; (c) loss function combination — only "softmax loss" implied |
| Architecture paper (background) | Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021 — [2103.14030_SwinTransformer_Liu2021.pdf](coursework/papers/2103.14030_SwinTransformer_Liu2021.pdf) — no VeRi-776 numbers; Swin-T reported as 81.3% top-1 on ImageNet-1K |
| Cross-validation (CLIP-SENet Table I, p. 6) | CLIP-SENet's SOTA table does **not** list Li et al. 2022 Swin-T as a baseline, so 78.6 / 97.3 is most directly comparable only to TransReID-era methods, not modern CLIP/IBN-based SOTA |
| Cross-validation (GLSIPNet Table 2, p. 10, [applsci-15-07041.pdf](coursework/papers/applsci-15-07041.pdf)) | Reproduces Li et al. 2022 number: "Swin Transformer [53] → 78.6 mAP / 97.3 R1" |
| Comparable VeRi-776 transformer numbers (TransReID Table 2 + Table 6) | DeiT-S/16 → 76.3 / 95.5 · DeiT-B/16 → 78.4 / 95.9 · ViT-B/16 → 78.2 / 96.5 · ViT-B/16 sliding s=12 → 79.0 / 96.5 · Full TransReID\* (DeiT-B/16, cam+view) → **82.3 / 97.1** |
| TransReID hyperparameters for transformer Re-ID on VeRi-776 (§4.2, p. 5-6) | Image 256×256, h-flip + pad + random crop + random erasing; batch 64 (4 imgs/ID); **SGD** mom=0.9, wd=1e-4; lr 0.008 cosine decay; FP16; ImageNet-21K → ImageNet-1K pretrain |
| Optimiser warning (TransReID Table 7, p. 11) | For transformer Re-ID, **SGD beats Adam by 12.4 mAP on VeRi-776** (78.2 vs 65.8). Our [main.py](coursework/main.py) defaults to Adam/AMSGrad — likely under-trains Swin/CLIP variants |

### 8.4 `mobilenet_v3_small`

| Field | Value |
|---|---|
| Paper | **Not present** in [papers/](coursework/papers/) (Howard et al., *Searching for MobileNetV3*, ICCV 2019, arXiv:1905.02244 — would need separate download) |
| VeRi-776 numbers in any folder paper | **None** — none of the 9 downloaded papers benchmark MobileNetV3 on VeRi-776 |
| Closest in-folder reference for MobileNetV3 on Re-ID | **None.** The MobileNetV3 paper reports ImageNet/COCO only; no Re-ID paper in the folder uses MobileNetV3 as backbone |
| Best published VeRi-776 number for any small/efficient backbone in folder | None reported — all CNN baselines in TransReID use ResNet/ResNeSt; CLIP-SENet uses ResNeXt-IBN |
| Implication for this coursework | `mobilenet_v3_small` should be treated as a **lightweight reference** for params/latency vs. accuracy trade-offs — there is no published VeRi-776 number to compare against. Numbers must come from your own runs |

### 8.5 GLSIPNet (additional ResNet-50 reference for VeRi-776)

| Field | Value |
|---|---|
| Paper | Nath & Mitra, *Learning Part-Based Features for Vehicle Re-Identification with Global Context*, **MDPI Applied Sciences 15 (2025) 7041** — [applsci-15-07041.pdf](coursework/papers/applsci-15-07041.pdf) |
| Backbone | **ResNet-50** ImageNet-pretrained (§3.2, p. 7) — directly comparable to local `resnet50_fc512` |
| Best VeRi-776 (Table 1, p. 10) | **GLSIPNet → 76.76 mAP / 94.63 R1 / 97.55 R5 / 98.92 R10**; with re-ranking → **80.99 / 95.70 / 97.55 / 98.68** |
| Baseline (PCB-style, no global similarity) on VeRi-776 (Table 1, p. 10) | 74.17 mAP / 94.04 R1 (+RR: 78.75 / 94.69) — provides another **ResNet-50 + horizontal-PCB baseline** number for VeRi-776 (cf. TransReID's 76.4 mAP, §8.1) |
| Special hyperparameters (§4, p. 9) | **Input 384×192**, **batch 32**, **6 horizontal parts** (PCB-style), **60 + 10 epochs** (cycle-1 part-only + cycle-2 with global similarity), augmentations = h-flip + random erasing + ImageNet normalisation, distance = Euclidean with similarity f(x)=1/x, **k-reciprocal re-ranking** as post-processing |
| Method (§3.3, p. 7-8) | Two training cycles: cycle 1 trains 6 horizontal parts independently with cross-entropy; cycle 2 weights each part's CE loss by its inverse Euclidean distance to the global feature vector — adds a *global signal* without adding trainable parameters |
| Compared methods on VeRi-776 (Table 2, p. 10) | Confirms Swin-T 78.6 / 97.3 (Li et al.) and TransReID 78.2 / 96.5; lists 18 methods total — useful sanity check for §8 numbers |

### 8.6 BoT tricks reference grid

| Trick | What it does | Paper § |
|---|---|---|
| **Warmup LR** | Start slow, ramp up — stops early large updates from breaking pretrained weights | §3.1 |
| **Random Erasing** | Randomly blacks out a patch — model can't rely on one part of the vehicle | §3.2 |
| **Label Smoothing** | Softens hard targets (ε=0.1) — stops the model becoming overconfident on training IDs | §3.3 |
| **Last-stride = 1** | Keeps more spatial detail in the final ResNet layer — helps spot fine-grained differences like plates and badges | §3.4 |
| **BNNeck** | Normalises before classification but not before distance comparison — separates the two tasks which need features in different shapes | §3.5 |
| **Center Loss** | Extra penalty if same-vehicle images end up far apart — pulls embeddings into tight clusters | §3.6 |
| **Hard Triplet** | Mines the examples the model is currently getting most wrong — harder examples mean faster learning | §3.6 |
| **Identity Sampler** | Guarantees each batch has multiple photos of the same vehicles — without this triplet loss rarely sees useful pairs | §2 |
| **Optimiser** | Adam lr=3.5e-4, drops ×0.1 at epochs 40 & 70 — big updates early, fine-tuning later, 120 epochs total | §2 |

### 8.7 Cross-paper recipe summary for VeRi-776

| Recipe element | BoT (person, p. 2-6) | TransReID (vehicle, §4.2 p. 5-6) | CLIP-SENet (vehicle, §IV.B p. 6-7) |
|---|---|---|---|
| Input H×W | 256×128 | 256×256 | 320×320 |
| Optimiser | Adam | SGD mom=0.9 wd=1e-4 | ADAM |
| Initial LR | 3.5e-4 | 8e-3 | 5e-4 |
| LR schedule | MultiStep ×0.1 @ {40, 70} + 10-ep linear warmup | Cosine decay | Cosine annealing |
| Batch / sampler | 64 (P=16, K=4) | 64 (4 imgs/ID, P=16) | 128 (P=16, K=8) |
| Epochs | 120 | not specified for VeRi (cosine to convergence) | **24** for VeRi-776 |
| Triplet margin | 0.3 | soft-margin (Eq. 3, p. 4) | not used — uses SupCon instead |
| Label smoothing | ε=0.1 | **off** (BoT-style smoothing harmful per Table 7, p. 11) | ε=0.1 |
| Random erasing | p=0.5 | yes | "various augmentation" |
| BNNeck | ✓ | ✓ | ✓ (BN + ReLU before AFEM) |
| Last stride = 1 | ✓ | n/a (transformer) | n/a (uses IBN ResNeXt) |
| Aux signal | Center loss β=5e-4 | Camera + viewpoint embedding (SIE, λ=2.5) | SupCon + camera/viewpoint embedding |
| Re-ranking on VeRi-776 | not used | not used | **applied** (only on VeRi-776) |

---

## 9. File map

| File | Purpose |
|---|---|
| [coursework/src/models/__init__.py](coursework/src/models/__init__.py) | Model factory (`get_names`, `init_model`) |
| [coursework/src/models/resnet.py](coursework/src/models/resnet.py) | ResNet-18/34/50 + `*_fc512` variants |
| [coursework/src/models/tvmodels.py](coursework/src/models/tvmodels.py) | torchvision `mobilenet_v3_small`, legacy `swin_t_fc512` |
| [coursework/src/models/timm_model.py](coursework/src/models/timm_model.py) | timm Swin-Tiny: `swin_t_fc512`, `swin_t_custom` |
| [coursework/src/models/clip_senet.py](coursework/src/models/clip_senet.py) | `clip_senet_v1_dual`, `clip_senet_v2_frozen` |
| [coursework/src/models/clip_senet_v3.py](coursework/src/models/clip_senet_v3.py) | `clip_senet_v3_ibn_supcon` (IBN + SupCon) |
| [coursework/main.py](coursework/main.py) | Train/eval loop, AMP, comparison summary export |
| [coursework/args.py](coursework/args.py) | CLI surface — every hyper-parameter listed in §5 |
| [coursework/train.sh](coursework/train.sh) | Active VeRi-776 recipe |

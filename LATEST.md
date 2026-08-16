## What this is
Vehicle re-identification (ReID) research/coursework repo targeting VeRi-776. Two strands: an original `reid` codebase with custom models/Colab notebooks, and an imported Surrey EEEM071 coursework codebase (`coursework/`) with dataset, model, loss, and training utilities, extended with new backbones and CLIP-based fusion models.

## Where it runs
Local dev on Windows (RTX 5070 Ti, CUDA 12.8 nightly torch) plus Colab notebooks (`colab_reid.ipynb`, `EEEM071_CourseWork2604_Colab.ipynb`) for GPU training; VeRi-776 data expected at `G:\My Drive\VeRi` locally or `/content/drive/MyDrive/Veri` on Colab. Not deployed as a service.

## Features
- Training/eval pipeline (`coursework/main.py`): bf16 autocast, xent + hard-batch triplet loss, MultiStep LR, DataParallel, mAP/CMC metrics.
- Model zoo: ResNet18/34/50 (+fc512), MobileNetV3-small, Swin-T (timm), DeiT-S, and CLIP-SENet v1-v4 (dual-stream ConvNeXt/ViT/CLIP fusion with AFEM + SupCon aux loss).
- Sweep scripts under `scripts/` (per-model train.sh, lr/aug/bs sweeps) and log parsers (`parse_logs.py`, `parse_coursework_logs.py`).
- Image exploration notebook/tools (`explore_images.ipynb`, `coursework/src/utils/explore.py`).
- Consolidated knowledge base with published-paper benchmarks in `coursework/knowledge.md`.

## Recently tried
- 2026-07-16: clip_senet v3 IBN+SupCon fixes, new v4 model, timm/tv model updates, added sweep scripts + log parsers (`20cd78c`).
- 2026-04-27: commit "dne" — refactors to `convert_224.py`/veri dataset/transforms (`0380102`).
- 2026-04-26: added torchreid dependency, then several "changes"/"checkin" commits refining the Colab notebook (`64a2fac`..`654ab54`).
- 2026-04-26: applied AMP (mixed precision) for faster Colab runs (`61faec7`).
- 2026-04-26: earlier "Simpler changes" / "lots changes" iterations on the Colab notebook (`1378a11`, `e82961e`).

## Next
- BoT/AGW parity gaps flagged in knowledge.md: enable `--label-smooth` and `--random-erase`, switch to `RandomIdentitySampler` (P×K), add LR warmup — cheapest wins toward published numbers.
- clip_senet_v4 needs full sweep runs (v3 already swept per commit history).
- (Inferred) consolidate the many exploratory notebooks/checkin commits once a stable training recipe is settled.

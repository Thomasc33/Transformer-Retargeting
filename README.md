# DisentangledTMR

### Privacy-Preserving Skeleton Motion Retargeting via Factorized Transformers

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://tmr.thomasc.tech)
[![Conference](https://img.shields.io/badge/ECCV-2026-purple)]()

Official implementation of **"DisentangledTMR: Privacy-Preserving Skeleton Motion
Retargeting via Factorized Transformers"**, accepted to **ECCV 2026**.

Thomas Carr<sup>1,2</sup>, Depeng Xu<sup>1</sup>, Shuhan Yuan<sup>3</sup>, Aidong Lu<sup>1</sup>
<br>
<sup>1</sup>University of North Carolina at Charlotte &nbsp;
<sup>2</sup>Incerta Intelligence &nbsp;
<sup>3</sup>Utah State University

---

## Overview

DisentangledTMR anonymizes 3D skeleton motion by transferring the *action* from one
person onto the *identity* (skeleton structure) of another — preserving what is being
done while changing who appears to be doing it. Skeleton sequences are 3D joint
positions over time, shaped `(B, C, T, V, M)` = (batch, 3 coords, 64 frames, 25 joints,
1 person).

The model factorizes motion into two disentangled latent streams:

- **Action encoder** — position/velocity/acceleration → multi-scale temporal convolution
  (kernels 3, 5, 7) → temporal attention → LSTM, with an optional MixFormer backbone fused
  through a learned gate. Captures motion dynamics.
- **Identity encoder** — static pose + bone lengths → spatial GCN with attention → a compact
  identity code. Low capacity by design, to limit identity leakage.
- **Factorized decoder** — separate cross-attention over the action and identity streams with
  an adaptive fusion gate (`fused = gate * action + (1 - gate) * identity`), autoregressive
  with causal masking and teacher forcing.

Disentanglement is enforced with contrastive (InfoNCE), adversarial (gradient reversal),
orthogonality, and mutual-information losses; reconstruction is regularized with physical
plausibility terms (bone length, temporal smoothness, velocity, end-effector, foot contact,
joint limits).

## Installation

Requires Python ≥ 3.9 and PyTorch.

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA/platform from
[pytorch.org](https://pytorch.org) if the default wheel does not fit your system.

## Data

Experiments use **NTU RGB+D 60/120** and **ETRI-Activity3D**. Preprocessing utilities for
each dataset are under `data/`. Paired training samples are stored as `.pt` files
containing `Cross_Data` objects (quadruplets of (identity, action) pairs for
cross-identity training). Model weights and datasets are not distributed with this
repository.

## Usage

The `tmr.py` CLI is the main entry point; it auto-detects local vs. SLURM environments.

```bash
# Three-stage training
python tmr.py train --data_path data/ntu_cv_paired.pt --dataset ntu \
  --output_dir output/disentangled_tmr

# Evaluate a checkpoint on downstream action/identity recognition
python tmr.py eval --checkpoint output/disentangled_tmr/checkpoint_stage3_best.pth \
  --data_path data/ntu_cv_paired.pt --dataset ntu

# Check experiment status
python tmr.py status

# Full pipeline: retarget -> train downstream models -> evaluate
python tmr.py run-pipeline
```

### Three-stage training protocol

Training proceeds in three stages and the order is mandatory — skipping stages breaks
disentanglement:

1. **Encoder pretraining** — train the action/identity encoders with classification heads
   and disentanglement losses; decoder frozen.
2. **Decoder training** — train the decoder with reconstruction + physical losses; encoders
   frozen. Teacher forcing 1.0 → 0.5.
3. **End-to-end fine-tuning** — all components; teacher forcing 0.5 → 0.3. Optional
   `--freeze_encoders_stage3` for a hybrid mode.

## Repository layout

- `tmr.py` — main CLI (train / eval / status / run-pipeline)
- `src/model/` — DisentangledTMR: action encoder, identity encoder, factorized decoder, and
  baseline backbones (SGN, Skeleton-MixFormer)
- `src/training/` — disentanglement losses and training utilities
- `src/data/` — dataset loaders and pairing (`Cross_Data`)
- `src/losses/` — physical plausibility and reconstruction losses
- `src/graph/` — NTU skeleton graph definition (25 joints, 24 bones)
- `scripts/` — training, evaluation, and dataset-generation entry points
- `configs/` — model and training configuration
- `eval/` — downstream action/identity recognition evaluation
- `data/` — dataset preprocessing utilities
- `index.html`, `assets/` — interactive project page (hosted at <https://tmr.thomasc.tech>)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{carr2026disentangledtmr,
  title     = {DisentangledTMR: Privacy-Preserving Skeleton Motion Retargeting via Factorized Transformers},
  author    = {Carr, Thomas and Xu, Depeng and Yuan, Shuhan and Lu, Aidong},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

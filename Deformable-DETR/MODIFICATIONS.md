# Modifications from Original Deformable-DETR

This document comprehensively summarizes all modifications made to the [original Deformable-DETR repository](https://github.com/fundamentalvision/Deformable-DETR/tree/main) in this project (CS776 Group 19).

---

## Table of Contents

1. [Overview](#overview)
2. [New Files Added](#new-files-added)
3. [Modified Files](#modified-files)
4. [Key Architectural Changes](#key-architectural-changes)
5. [New Features Implemented](#new-features-implemented)
6. [Training / Inference Modifications](#training--inference-modifications)
7. [Loss Functions and Training Logic Changes](#loss-functions-and-training-logic-changes)
8. [Utility Functions and Helper Code Added](#utility-functions-and-helper-code-added)
9. [Summary Table](#summary-table)

---

## Overview

The core objective of this project is to extend Deformable-DETR with **aleatoric uncertainty estimation** for bounding boxes — sometimes referred to as "Track A" in the codebase. The model learns to predict a **log-variance** for each spatial dimension of each predicted bounding box, enabling it to express how uncertain it is about each detection.

Key capabilities added:

- A dedicated **variance head** (`var_embed`) attached to the decoder.
- A numerically stable **heteroscedastic NLL loss** (`loss_track_a`) that jointly trains detection quality and uncertainty calibration.
- A **fine-tuning mode** that freezes the entire pre-trained detector and trains only the new variance head.
- Enhanced checkpoint loading that is backward-compatible with checkpoints that do not contain the variance head.

---

## New Files Added

### 1. `download_checkpoint.py` (project root)

**Purpose:** Provides a convenient script to download official pre-trained Deformable-DETR checkpoints from Google Drive.

**Key details:**
- Supports three model variants:
  - `deformable_detr` — 50-epoch ResNet-50 baseline (AP 44.5 %)
  - `deformable_detr_plus` — with iterative bounding box refinement (AP 46.2 %)
  - `deformable_detr_pp` — with iterative refinement + two-stage proposal (AP 46.9 %)
- Uses `gdown` to download from Google Drive URLs.
- Automatically creates the `Deformable-DETR/models/` directory if it does not exist.
- Selects which model to download via a command-line argument.

**Why added:** Makes it easy to obtain the official pre-trained weights without manually navigating Google Drive links.

---

### 2. `verify_checkpoint.py` (project root)

**Purpose:** Loads a checkpoint file and prints a structured summary of its contents, along with official baseline performance numbers.

**Key details:**
- Displays file path and size (MB).
- Loads the checkpoint with `weights_only=False` (required for PyTorch ≥ 2.6 to load legacy checkpoints that contain arbitrary Python objects).
- Prints the number of stored model parameters, the training epoch, and whether optimizer / LR-scheduler state is included.
- Prints the official COCO val2017 performance table.
- Notes the CUDA compilation requirement and suggests next steps.

**Why added:** Helps quickly audit checkpoints and confirm training epoch/quality without running full inference.

---

### 3. `SETUP_COMPLETE.md` (project root)

**Purpose:** Project-level setup documentation covering environment creation, library installation, checkpoint download status, model variants, and quick-start commands.

**Why added:** Serves as a self-contained onboarding guide for the project, recording what was installed and how to reproduce the setup on any machine.

---

## Modified Files

### 1. `Deformable-DETR/models/deformable_detr.py`

This is the most heavily modified file. All changes relative to the upstream version are listed below.

#### A. `DeformableDETR.__init__` — new constructor parameters

```python
# Original signature
def __init__(self, backbone, transformer, num_classes, num_queries,
             num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False):

# Modified signature
def __init__(self, backbone, transformer, num_classes, num_queries,
             num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False,
             with_track_a=False, log_var_min=-8.0, log_var_max=8.0):
```

New parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_track_a` | `bool` | `False` | Enable the variance head and aleatoric loss |
| `log_var_min` | `float` | `-8.0` | Lower clamp for predicted log-variance (during forward pass) |
| `log_var_max` | `float` | `8.0` | Upper clamp for predicted log-variance (during forward pass) |

These values are saved as instance attributes (`self.with_track_a`, `self.log_var_min`, `self.log_var_max`).

#### B. `DeformableDETR.__init__` — variance head (`var_embed`)

```python
# Added: variance prediction MLP
if self.with_track_a:
    self.var_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    nn.init.constant_(self.var_embed.layers[-1].weight.data, 0)
    nn.init.constant_(self.var_embed.layers[-1].bias.data, 0)
```

- `var_embed` has the same architecture as `bbox_embed`: a 3-layer MLP with hidden size equal to the transformer hidden dimension, producing a 4-dimensional output (one log-variance per box coordinate: `cx, cy, w, h`).
- Weights and biases of the last layer are initialized to zero so the head starts with no predicted uncertainty, letting the loss drive calibration.

When `with_box_refine=True`, `var_embed` is cloned per decoder layer (same as `bbox_embed`):

```python
if with_box_refine:
    ...
    if self.with_track_a:
        self.var_embed = _get_clones(self.var_embed, num_pred)
```

When `with_box_refine=False`, the same single `var_embed` module is reused across all decoder layers (matching the behavior of `bbox_embed` and `class_embed`):

```python
else:
    ...
    if self.with_track_a:
        self.var_embed = nn.ModuleList([self.var_embed for _ in range(num_pred)])
```

#### C. `DeformableDETR.forward` — variance prediction in decoder loop

```python
# Inside the per-layer decoder loop:
if self.with_track_a:
    outputs_log_var = self.var_embed[lvl](hs[lvl])
    outputs_log_var = torch.clamp(outputs_log_var, min=self.log_var_min, max=self.log_var_max)
    outputs_log_vars.append(outputs_log_var)
```

- The list `outputs_log_vars` is initialized before the loop.
- For each decoder layer `lvl`, the variance head is applied to the decoder hidden states.
- Log-variances are immediately clamped within `[log_var_min, log_var_max]` to prevent runaway outputs from an untrained head.

After the loop:

```python
out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
if self.with_track_a:
    outputs_log_var = torch.stack(outputs_log_vars)
    out['pred_log_vars'] = outputs_log_var[-1]
```

The final-layer log-variance is added to the model output dictionary under `'pred_log_vars'`.

#### D. `DeformableDETR._set_aux_loss` — propagate log-vars to auxiliary outputs

```python
# Original
@torch.jit.unused
def _set_aux_loss(self, outputs_class, outputs_coord):
    return [{'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

# Modified
@torch.jit.unused
def _set_aux_loss(self, outputs_class, outputs_coord, outputs_log_var=None):
    if outputs_log_var is None:
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    return [
        {'pred_logits': a, 'pred_boxes': b, 'pred_log_vars': c}
        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_log_var[:-1])
    ]
```

This ensures that each intermediate decoder layer also outputs `pred_log_vars`, enabling the auxiliary loss to supervise uncertainty at every layer.

#### E. `SetCriterion.get_loss` — register `track_a` loss

```python
loss_map = {
    'labels':      self.loss_labels,
    'cardinality': self.loss_cardinality,
    'boxes':       self.loss_boxes,
    'masks':       self.loss_masks,
    'track_a':     self.loss_track_a,   # ← new
}
```

#### F. `SetCriterion.loss_track_a` — new heteroscedastic uncertainty loss

```python
def loss_track_a(self, outputs, targets, indices, num_boxes):
    ...
```

See [Loss Functions and Training Logic Changes](#loss-functions-and-training-logic-changes) for the full description.

#### G. `build` function — wire new arguments and loss weights

```python
model = DeformableDETR(
    ...
    with_track_a=args.track_a,
    log_var_min=args.log_var_min,
    log_var_max=args.log_var_max,
)

weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
weight_dict['loss_giou'] = args.giou_loss_coef
if args.track_a:
    weight_dict['loss_track_a'] = args.track_a_loss_coef   # ← new

losses = ['labels', 'boxes', 'cardinality']
if args.track_a:
    losses += ['track_a']   # ← new
```

---

### 2. `Deformable-DETR/main.py`

#### A. New command-line arguments

Three new arguments added to `get_args_parser()`:

```python
parser.add_argument('--track_a_loss_coef', default=1.0, type=float,
                    help="Loss weight for Track A heteroscedastic bbox term")
parser.add_argument('--track_a', action='store_true',
                    help="Enable Track A variance head and aleatoric loss")
parser.add_argument('--log_var_min', default=-8.0, type=float,
                    help="Minimum clamp value for predicted log-variance")
parser.add_argument('--log_var_max', default=8.0, type=float,
                    help="Maximum clamp value for predicted log-variance")
```

#### B. Selective parameter freezing for fine-tuning

```python
if args.track_a:
    # Track A fine-tuning: freeze the baseline detector and train only variance head(s).
    for name, param in model.named_parameters():
        param.requires_grad = 'var_embed' in name
```

When `--track_a` is passed, **all parameters except `var_embed`** are frozen (`requires_grad=False`). This implements the design goal of loading a fully-trained detection checkpoint and fine-tuning only the uncertainty estimation head.

#### C. Robust checkpoint loading with optimizer compatibility

The original code assumed that the optimizer state in the checkpoint was compatible with the current model. After adding `var_embed`, the optimizer may contain parameter groups that do not match a newly instantiated model. A `try/except` block was added:

```python
try:
    optimizer.load_state_dict(checkpoint['optimizer'])
    for pg, pg_old in zip(optimizer.param_groups, p_groups):
        pg['lr'] = pg_old['lr']
        pg['initial_lr'] = pg_old['initial_lr']
    ...
    args.start_epoch = checkpoint['epoch'] + 1
except ValueError as e:
    if utils.is_main_process():
        print(f"WARNING: Could not load optimizer state (new parameters detected): {e}")
        print("Starting with fresh optimizer for new var_embed parameters")
    args.start_epoch = 0
```

This allows fine-tuning to start cleanly even when resuming from a base checkpoint that pre-dates the variance head.

#### D. Evaluation run after checkpoint loading

```python
if args.resume:
    ...
    # check the resumed model
    if not args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
```

An evaluation pass is run immediately after loading the checkpoint (and before any new training epoch). This provides an "epoch 0 / baseline" measurement to confirm that the loaded weights are intact and that the variance head does not degrade detection performance.

#### E. `weights_only=False` for checkpoint loading

```python
checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
```

Added `weights_only=False` to maintain compatibility with legacy checkpoints saved with arbitrary Python objects (e.g., the `args` namespace) in PyTorch ≥ 2.6.

---

## Key Architectural Changes

### Variance Head (`var_embed`)

```
Decoder Hidden State  →  var_embed (MLP, 3 layers)  →  log_var (4-dim)
                                                             ↓
                                               clamp([log_var_min, log_var_max])
                                                             ↓
                                                    pred_log_vars output
```

- **Architecture:** Identical to the bbox regression head (`bbox_embed`): a 3-layer MLP with ReLU activations, hidden dimension = transformer hidden dimension (256 by default), output dimension = 4.
- **Initialization:** Last-layer weights and biases are zeroed, so the head starts by predicting zero log-variance (unit variance = 1.0), meaning "maximally uncertain but not unreasonably so".
- **Clamping:** Applied at two stages:
  1. **Forward pass** — broad clamp `[log_var_min, log_var_max]` (default `[-8.0, 8.0]`) to prevent numerical overflow.
  2. **Loss computation** — tighter clamp `[-4.0, 4.0]` (hardcoded in `loss_track_a`) to prevent the untrained head from driving the loss to infinity in early training.
- **Scope:** Available both in standard and `with_box_refine` modes. In both modes it mirrors the `bbox_embed` structure.

---

## New Features Implemented

### Aleatoric Uncertainty Estimation (Track A)

Aleatoric uncertainty refers to irreducible noise in the data — e.g., occluded objects or blur. By predicting a per-box log-variance, the model can express:

- **High confidence:** small variance → tight detection.
- **Low confidence:** large variance → uncertain detection, should be treated with skepticism downstream.

This is useful for:

- Downstream tasks that need calibrated confidence (e.g., tracking, active learning).
- Identifying hard examples during analysis.
- Potentially improving AP by reducing overconfident false positives.

---

## Training / Inference Modifications

### Training Mode (`--track_a`)

| Aspect | Original | Modified (with `--track_a`) |
|--------|----------|-----------------------------|
| Trainable parameters | All | Only `var_embed` layers |
| Loss terms | `loss_ce`, `loss_bbox`, `loss_giou` | All above + `loss_track_a` |
| Requires base checkpoint | No | Yes (detector is frozen) |
| Output dict | `pred_logits`, `pred_boxes` | + `pred_log_vars` |
| Auxiliary outputs | `pred_logits`, `pred_boxes` per layer | + `pred_log_vars` per layer |

### Inference

No changes to the inference / evaluation pipeline in `engine.py`. The `PostProcess` module is unchanged. The extra `pred_log_vars` tensor in the model output is silently ignored during standard COCO evaluation.

---

## Loss Functions and Training Logic Changes

### `loss_track_a` — Heteroscedastic NLL Loss

**Mathematical form:**

```
L = (error² / var) + log(var)
  = (src_box - tgt_box)² · exp(-log_var) + log_var
```

where `error` is the element-wise squared difference between the predicted and target box coordinates (in normalized `[0,1]` space), and `var = exp(log_var)`.

**Derivation:** This is the standard aleatoric uncertainty objective from Kendall & Gal (2017), *"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"*. The loss is a negative log-likelihood under a Gaussian noise model:

```
p(y | x, σ²) = N(f(x), σ²)
NLL = 0.5 * log(2π σ²) + (y - f(x))² / (2σ²)
    ≈ log_var + error² / exp(log_var)   [up to constants and 0.5 factor]
```

The 0.5 factor is omitted to make hyperparameter tuning of `track_a_loss_coef` more intuitive.

**Numerical stability measures:**

1. **Tight clamping of `log_var` (inner):** Before computing `exp(log_var)`, the raw `src_log_vars` are clamped to `[-4.0, 4.0]`. This bounds `var ∈ [0.018, 54.6]`, preventing division by near-zero or multiplication by near-infinity.

2. **Loss clamping:** The element-wise loss is clamped to `[0.0, 100.0]` to guard against any remaining extreme values.

3. **NaN guard #1 (element-wise):** After clamping, if any element is still `NaN`, the function prints diagnostics and returns `loss_track_a = 0.0` (with `requires_grad=True` to avoid graph issues).

4. **NaN guard #2 (aggregate):** If the mean loss is `NaN` or `Inf`, the same zero-loss fallback is used.

**Why MSE instead of L1?**

The loss uses `(src - tgt)² ` (L2/MSE) rather than `|src - tgt|` (L1). L2 provides smoother gradients near the optimum and is the natural error term for a Gaussian likelihood model, making the math consistent.

**Integration with auxiliary losses:**

The `track_a` loss is applied at every decoder layer via the existing auxiliary-loss machinery in `SetCriterion.forward`. Because `pred_log_vars` is now included in `aux_outputs`, each layer's uncertainty head is supervised independently.

---

## Utility Functions and Helper Code Added

### `compare_checkpoints.py` (project root)

A standalone script that compares two checkpoint files — for example, a baseline checkpoint (`checkpoint0000.pth`) and a fine-tuned checkpoint (`checkpoint.pth`) — and reports:

- File sizes
- Last-modified timestamps
- Which file is more recent
- Training epoch stored in each checkpoint
- Number of model parameters in each checkpoint
- Whether the newer checkpoint contains the new `var_embed` parameters

Usage:

```bash
python compare_checkpoints.py \
    /path/to/checkpoint0000.pth \
    /path/to/checkpoint.pth
```

This utility resolves the ambiguity users experience when trying to determine which checkpoint is the "latest" finetuned model.

---

## Summary Table

| # | Change | File(s) | Category |
|---|--------|---------|----------|
| 1 | Added `var_embed` MLP head for log-variance prediction | `models/deformable_detr.py` | Architecture |
| 2 | Cloned `var_embed` per decoder layer when `with_box_refine=True` | `models/deformable_detr.py` | Architecture |
| 3 | Clamped log-variance in forward pass | `models/deformable_detr.py` | Architecture |
| 4 | Added `pred_log_vars` to model output dict | `models/deformable_detr.py` | Architecture |
| 5 | Extended `_set_aux_loss` to propagate log-vars to auxiliary outputs | `models/deformable_detr.py` | Architecture |
| 6 | Added `loss_track_a` heteroscedastic NLL loss | `models/deformable_detr.py` | Loss Function |
| 7 | Registered `track_a` in `SetCriterion.get_loss` | `models/deformable_detr.py` | Loss Function |
| 8 | Added `loss_track_a` to `weight_dict` in `build()` | `models/deformable_detr.py` | Loss Function |
| 9 | Added `--track_a`, `--track_a_loss_coef`, `--log_var_min`, `--log_var_max` CLI args | `main.py` | Training Config |
| 10 | Selective parameter freezing (only `var_embed` trains when `--track_a`) | `main.py` | Training Logic |
| 11 | Robust checkpoint loading (handles optimizer mismatch with `try/except`) | `main.py` | Training Logic |
| 12 | Evaluation pass immediately after checkpoint loading | `main.py` | Training Logic |
| 13 | Added `weights_only=False` for legacy checkpoint compatibility | `main.py` | Training Logic |
| 14 | New `download_checkpoint.py` utility | `download_checkpoint.py` | Utility |
| 15 | New `verify_checkpoint.py` utility | `verify_checkpoint.py` | Utility |
| 16 | New `compare_checkpoints.py` utility | `compare_checkpoints.py` | Utility |
| 17 | `SETUP_COMPLETE.md` project documentation | `SETUP_COMPLETE.md` | Documentation |
| 18 | `MODIFICATIONS.md` (this file) | `Deformable-DETR/MODIFICATIONS.md` | Documentation |

---

## References

- Zhu et al., *"Deformable DETR: Deformable Transformers for End-to-End Object Detection"*, ICLR 2021. ([arXiv](https://arxiv.org/abs/2010.04159))
- Kendall & Gal, *"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"*, NIPS 2017. ([arXiv](https://arxiv.org/abs/1703.04977))
- Original Deformable-DETR repository: https://github.com/fundamentalvision/Deformable-DETR

# Deformable-DETR: Detailed File Comparison

> **Comparing** `krishna-kumar-bais/CS776_Project` (this repo) vs
> `fundamentalvision/Deformable-DETR` (upstream, commit `main`)

---

## Summary of Changes

| File | Status |
|------|--------|
| `main.py` | **Modified** – Track A fine-tuning args + robust checkpoint loading |
| `models/deformable_detr.py` | **Modified** – Aleatoric uncertainty head (`var_embed`) + `loss_track_a` |
| `util/misc.py` | **Modified** – Version detection uses `packaging.version` instead of float cast |
| `engine.py` | **Unchanged** |
| `models/backbone.py` | **Unchanged** |

---

## 1. `main.py`

### 1a. New CLI arguments (get_args_parser)

**BEFORE (original):**
```python
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
```

**AFTER (this repo):**
```python
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--track_a_loss_coef', default=1.0, type=float,
                        help="Loss weight for Track A heteroscedastic bbox term")
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--track_a', action='store_true',
                        help="Enable Track A variance head and aleatoric loss")
    parser.add_argument('--log_var_min', default=-8.0, type=float,
                        help="Minimum clamp value for predicted log-variance")
    parser.add_argument('--log_var_max', default=8.0, type=float,
                        help="Maximum clamp value for predicted log-variance")
```

**What changed:**
- ➕ Added `--track_a_loss_coef` (default `1.0`): weight for the new aleatoric loss term
- ➕ Added `--track_a` (bool flag): enables the uncertainty head and `loss_track_a`
- ➕ Added `--log_var_min` (default `-8.0`): lower clamp bound for predicted log-variance
- ➕ Added `--log_var_max` (default `8.0`): upper clamp bound for predicted log-variance

---

### 1b. Track A parameter freezing (main function)

**BEFORE (original):**
```python
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
```

**AFTER (this repo):**
```python
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.track_a:
        # Track A fine-tuning: freeze the baseline detector and train only variance head(s).
        for name, param in model.named_parameters():
            param.requires_grad = 'var_embed' in name

    model_without_ddp = model
```

**What changed:**
- ➕ Added Track A freeze block: when `--track_a` is set, all parameters except `var_embed` are frozen so only the new variance head is trained during fine-tuning.

---

### 1c. Checkpoint loading – `weights_only` flag

**BEFORE (original):**
```python
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
```

**AFTER (this repo):**
```python
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
```

**What changed:**
- ➕ Added `weights_only=False` to suppress the `FutureWarning` introduced in PyTorch ≥ 2.0 and ensure the full checkpoint (including non-tensor objects like `args`) loads correctly.

---

### 1d. Robust optimizer state loading

**BEFORE (original):**
```python
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment...
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) ...')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
```

**AFTER (this repo):**
```python
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                print(optimizer.param_groups)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # todo: this is a hack for doing experiment...
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) ...')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                lr_scheduler.step(lr_scheduler.last_epoch)
                args.start_epoch = checkpoint['epoch'] + 1
            except ValueError as e:
                if utils.is_main_process():
                    print(f"WARNING: Could not load optimizer state (new parameters detected): {e}")
                    print("Starting with fresh optimizer for new var_embed parameters")
                args.start_epoch = 0
```

**What changed:**
- ➕ Wrapped optimizer/scheduler loading in `try/except ValueError`: when resuming a base checkpoint to fine-tune with the new `var_embed` head, the optimizer state has a different parameter count. The exception is caught gracefully, starting with a fresh optimizer instead of crashing.

---

## 2. `models/deformable_detr.py`

### 2a. DeformableDETR constructor – new parameters

**BEFORE (original):**
```python
class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        ...
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
```

**AFTER (this repo):**
```python
class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 with_track_a=False, log_var_min=-8.0, log_var_max=8.0):
        ...
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.with_track_a = with_track_a
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.with_track_a:
            self.var_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            nn.init.constant_(self.var_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.var_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if self.with_track_a:
                self.var_embed = _get_clones(self.var_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.with_track_a:
                self.var_embed = nn.ModuleList([self.var_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
```

**What changed:**
- ➕ New constructor params: `with_track_a=False`, `log_var_min=-8.0`, `log_var_max=8.0`
- ➕ New attributes: `self.with_track_a`, `self.log_var_min`, `self.log_var_max`
- ➕ Conditionally creates `self.var_embed = MLP(hidden_dim, hidden_dim, 4, 3)` (the aleatoric uncertainty head) with zero initialization
- ➕ In `with_box_refine` branch: `var_embed` is cloned `num_pred` times (one per decoder layer)
- ➕ In else branch: `var_embed` is wrapped in `nn.ModuleList`

---

### 2b. DeformableDETR.forward – uncertainty output

**BEFORE (original):**
```python
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
```

**AFTER (this repo):**
```python
        outputs_classes = []
        outputs_coords = []
        outputs_log_vars = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.with_track_a:
                outputs_log_var = self.var_embed[lvl](hs[lvl])
                outputs_log_var = torch.clamp(outputs_log_var, min=self.log_var_min, max=self.log_var_max)
                outputs_log_vars.append(outputs_log_var)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.with_track_a:
            outputs_log_var = torch.stack(outputs_log_vars)
            out['pred_log_vars'] = outputs_log_var[-1]
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_log_var if self.with_track_a else None
            )
```

**What changed:**
- ➕ Added `outputs_log_vars = []` accumulator list
- ➕ Per-decoder-layer: runs `var_embed[lvl]` and clamps with `[log_var_min, log_var_max]`
- ➕ Stacks and adds `pred_log_vars` to the output dict when `with_track_a`
- ➕ Passes `outputs_log_var` (or `None`) to `_set_aux_loss`

---

### 2c. DeformableDETR._set_aux_loss – optional log-variance

**BEFORE (original):**
```python
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
```

**AFTER (this repo):**
```python
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_log_var=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_log_var is None:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return [
            {'pred_logits': a, 'pred_boxes': b, 'pred_log_vars': c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_log_var[:-1])
        ]
```

**What changed:**
- ➕ Added optional `outputs_log_var=None` argument
- ➕ When `outputs_log_var` is provided, each auxiliary output dict includes `pred_log_vars`

---

### 2d. SetCriterion – new loss_track_a method

**BEFORE (original):** *(method did not exist)*

**AFTER (this repo):**
```python
    def loss_track_a(self, outputs, targets, indices, num_boxes):
        """Heteroscedastic uncertainty loss (MSE-based, numerically stable)
        
        Loss = error² / var + log(var)
        
        This is the standard form for aleatoric uncertainty quantification.
        Key improvements:
        - Uses MSE (L2) instead of L1 for better gradients
        - Uses error²/var + log(var) instead of exp(-log_var)*error + log_var
        - Tight clamping prevents numerical explosion
        - NaN guards catch any remaining issues
        """
        assert 'pred_boxes' in outputs
        assert 'pred_log_vars' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        src_log_vars = outputs['pred_log_vars'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Compute L2 error (MSE gives better gradients than L1)
        error = (src_boxes - target_boxes) ** 2
        
        # CRITICAL: Tight clamping BEFORE computing loss
        # This bounds log_var ∈ [-4, 4] → variance ∈ [0.018, 54.6]
        # Prevents untrained head from outputting extreme values
        log_var_clamped = torch.clamp(src_log_vars, min=-4.0, max=4.0)
        
        # Compute variance (now guaranteed to be in safe range)
        var = torch.exp(log_var_clamped)
        
        # Numerically stable heteroscedastic NLL loss
        # Standard form: Loss = 0.5 * (error² / var + log(var))
        # We omit the 0.5 factor for easier hyperparameter tuning
        nll_loss = error / var + log_var_clamped
        
        # Extra safety: clamp loss to prevent extreme values
        nll_loss = torch.clamp(nll_loss, min=0.0, max=100.0)
        
        # NaN guard #1: Check element-wise for NaN
        if torch.isnan(nll_loss).any():
            print(f"WARNING: NaN detected in Track A loss (element-wise)")
            ...
            return {'loss_track_a': torch.tensor(0.0, device=src_boxes.device, requires_grad=True)}
        
        # Aggregate loss
        loss_value = nll_loss.mean()
        
        # NaN guard #2: Check final loss
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            ...
            return {'loss_track_a': torch.tensor(0.0, device=src_boxes.device, requires_grad=True)}
        
        losses = {'loss_track_a': loss_value}
        return losses
```

**What changed:**
- ➕ Entirely new `loss_track_a` method implementing heteroscedastic NLL loss:
  - Computes squared error between predicted and target boxes
  - Applies tighter log-var clamping (`[-4, 4]`) inside the loss for numerical stability
  - Computes `nll_loss = error / var + log_var_clamped`
  - Guards against NaN/Inf with early returns

---

### 2e. SetCriterion.get_loss – register new loss

**BEFORE (original):**
```python
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
```

**AFTER (this repo):**
```python
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'track_a': self.loss_track_a,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
```

**What changed:**
- ➕ Added `'track_a': self.loss_track_a` entry to the loss dispatch map

---

### 2f. build() – wire up new components

**BEFORE (original):**
```python
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    ...
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    ...
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
```

**AFTER (this repo):**
```python
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_track_a=args.track_a,
        log_var_min=args.log_var_min,
        log_var_max=args.log_var_max,
    )
    ...
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.track_a:
        weight_dict['loss_track_a'] = args.track_a_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    ...
    losses = ['labels', 'boxes', 'cardinality']
    if args.track_a:
        losses += ['track_a']
    if args.masks:
        losses += ["masks"]
```

**What changed:**
- ➕ Passes `with_track_a`, `log_var_min`, `log_var_max` to `DeformableDETR`
- ➕ Conditionally adds `loss_track_a` weight to `weight_dict`
- ➕ Conditionally adds `'track_a'` to the losses list

---

## 3. `util/misc.py`

### 3a. torchvision version detection – float cast → packaging.version

**BEFORE (original):**
```python
import torchvision
if float(torchvision.__version__[:3]) < 0.5:
    import math
    from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        ...
    def _output_size(dim, input, size, scale_factor):
        ...
elif float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
```

...and later in `interpolate()`:

```python
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    ...
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(...)
        ...
        if float(torchvision.__version__[:3]) < 0.5:
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(...)
```

**AFTER (this repo):**
```python
import torchvision
from packaging import version
_TV_VERSION = version.parse(torchvision.__version__.split('+')[0])
if _TV_VERSION < version.parse("0.5"):
    import math
    from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        ...
    def _output_size(dim, input, size, scale_factor):
        ...
elif _TV_VERSION < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
```

...and later in `interpolate()`:

```python
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    ...
    if _TV_VERSION < version.parse("0.7"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(...)
        ...
        if _TV_VERSION < version.parse("0.5"):
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(...)
```

**What changed:**
- ➕ Added `from packaging import version` import
- ➕ Added module-level `_TV_VERSION = version.parse(torchvision.__version__.split('+')[0])`
- 🔄 All `float(torchvision.__version__[:3])` comparisons replaced with `_TV_VERSION < version.parse("X.Y")` — this is **more robust** for version strings like `0.15.0+cu118` where `__version__[:3]` extracts `"0.1"` (only 3 characters: `"0"`, `"."`, `"1"`) instead of the intended `"0.15"`, so `float("0.1") = 0.1 < 0.7` would incorrectly match the old-code path even though the actual version is `0.15` (which is `> 0.7`)

---

## 4. `engine.py` — No Changes

The file is **identical** to the upstream original.

---

## 5. `models/backbone.py` — No Changes

The file is **identical** to the upstream original.

---

## Overall Purpose of Modifications

The changes implement **Track A: Aleatoric Uncertainty Estimation** on top of Deformable-DETR:

| Component | Purpose |
|-----------|---------|
| `var_embed` MLP | Predicts per-box log-variance (4 dimensions matching bbox) |
| `loss_track_a` | Heteroscedastic NLL loss: `error²/var + log(var)` |
| `--track_a` flag | Enables uncertainty mode: adds head, adds loss term |
| `--track_a_loss_coef` | Controls weight of the new loss relative to detection losses |
| `--log_var_min/max` | Outer clamp on model output to prevent instability during training |
| Freeze logic | During fine-tuning, freezes the pretrained detector, trains only `var_embed` |
| Robust loading | `weights_only=False` + ValueError catch enable resuming from base checkpoints |
| Packaging version check | Prevents wrong comparisons for modern torchvision version strings |

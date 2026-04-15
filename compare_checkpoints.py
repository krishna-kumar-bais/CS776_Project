#!/usr/bin/env python3
"""
compare_checkpoints.py
======================
Compare two Deformable-DETR checkpoint files and determine which is more
recent and what parameters they contain.

Usage
-----
    python compare_checkpoints.py <checkpoint_a> <checkpoint_b>

Examples
--------
    python compare_checkpoints.py \\
        /kaggle/working/CV_Project/Deformable-DETR/models/checkpoint0000.pth \\
        /kaggle/working/CV_Project/Deformable-DETR/models/checkpoint.pth

When called with no arguments the script falls back to well-known default
paths used during the CS776 project training runs on Kaggle.
"""

import os
import sys
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_info(path: str) -> dict:
    """Return basic filesystem metadata for *path* without loading the file."""
    if not os.path.exists(path):
        return {"exists": False}
    stat = os.stat(path)
    return {
        "exists": True,
        "size_mb": stat.st_size / (1024 * 1024),
        "mtime": stat.st_mtime,
        "mtime_str": time.ctime(stat.st_mtime),
    }


def _checkpoint_info(path: str) -> dict:
    """Load a checkpoint with torch and extract training metadata."""
    import torch  # imported lazily so the script is importable without torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"load_error": str(exc)}

    info = {}

    if isinstance(ckpt, dict):
        info["keys"] = list(ckpt.keys())

        # Training epoch
        info["epoch"] = ckpt.get("epoch", "N/A")

        # Count model parameters
        model_state = ckpt.get("model", ckpt)
        if isinstance(model_state, dict):
            info["num_params"] = len(model_state)
            param_names = list(model_state.keys())
            info["has_var_embed"] = any("var_embed" in k for k in param_names)
            info["has_optimizer"] = "optimizer" in ckpt
            info["has_lr_scheduler"] = "lr_scheduler" in ckpt
        else:
            info["num_params"] = "unknown"
            info["has_var_embed"] = False

        # Training args (if present)
        if "args" in ckpt:
            args = ckpt["args"]
            info["track_a"] = getattr(args, "track_a", False)
            info["backbone"] = getattr(args, "backbone", "N/A")
            info["num_queries"] = getattr(args, "num_queries", "N/A")
    else:
        info["type"] = str(type(ckpt))

    return info


def _print_single(label: str, path: str, finfo: dict, cinfo: dict) -> None:
    print(f"\n  [{label}]  {path}")
    print(f"  {'─' * 60}")

    if not finfo.get("exists"):
        print("  ❌  File not found")
        return

    print(f"  Size        : {finfo['size_mb']:.1f} MB")
    print(f"  Modified    : {finfo['mtime_str']}")

    if "load_error" in cinfo:
        print(f"  Load error  : {cinfo['load_error']}")
        return

    print(f"  Epoch       : {cinfo.get('epoch', 'N/A')}")
    print(f"  Parameters  : {cinfo.get('num_params', 'N/A')}")
    print(f"  var_embed   : {'✅ YES (variance head present)' if cinfo.get('has_var_embed') else '❌ NO'}")
    print(f"  Optimizer   : {'yes' if cinfo.get('has_optimizer') else 'no'}")
    print(f"  LR scheduler: {'yes' if cinfo.get('has_lr_scheduler') else 'no'}")
    if cinfo.get("track_a") is not None:
        print(f"  track_a flag: {cinfo['track_a']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_checkpoints(path_a: str, path_b: str) -> None:
    print("=" * 70)
    print("🔍  CHECKPOINT COMPARISON")
    print("=" * 70)

    finfo_a = _file_info(path_a)
    finfo_b = _file_info(path_b)

    # Load checkpoint metadata only if the file exists
    cinfo_a = _checkpoint_info(path_a) if finfo_a.get("exists") else {}
    cinfo_b = _checkpoint_info(path_b) if finfo_b.get("exists") else {}

    _print_single("A", path_a, finfo_a, cinfo_a)
    _print_single("B", path_b, finfo_b, cinfo_b)

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📋  VERDICT")
    print("=" * 70)

    both_exist = finfo_a.get("exists") and finfo_b.get("exists")

    if not both_exist:
        if not finfo_a.get("exists"):
            print(f"  ❌  File A not found: {path_a}")
        if not finfo_b.get("exists"):
            print(f"  ❌  File B not found: {path_b}")
        return

    # Which is newer by filesystem timestamp?
    if finfo_a["mtime"] > finfo_b["mtime"]:
        newer_label, newer_path = "A", path_a
        older_label, older_path = "B", path_b
    elif finfo_b["mtime"] > finfo_a["mtime"]:
        newer_label, newer_path = "B", path_b
        older_label, older_path = "A", path_a
    else:
        newer_label = newer_path = None  # identical timestamps

    if newer_label:
        print(f"\n  ✅  LATEST checkpoint  →  [{newer_label}]")
        print(f"       {newer_path}")
        print(f"\n  🕒  Modified: {finfo_a['mtime_str'] if newer_label == 'A' else finfo_b['mtime_str']}")
        print(f"\n  ⏳  [{older_label}] is OLDER:")
        print(f"       {older_path}")
    else:
        print("\n  ⚠️  Both files have the same modification timestamp.")

    # Which has a higher training epoch?
    epoch_a = cinfo_a.get("epoch", None)
    epoch_b = cinfo_b.get("epoch", None)
    try:
        epoch_a_int = int(epoch_a)
        epoch_b_int = int(epoch_b)
        if epoch_a_int > epoch_b_int:
            print(f"\n  📈  [A] was trained further (epoch {epoch_a_int} vs {epoch_b_int})")
        elif epoch_b_int > epoch_a_int:
            print(f"\n  📈  [B] was trained further (epoch {epoch_b_int} vs {epoch_a_int})")
        else:
            print(f"\n  📈  Both at the same epoch ({epoch_a_int})")
    except (TypeError, ValueError):
        pass  # epoch not available or not an integer

    # Variance head presence
    va = cinfo_a.get("has_var_embed", False)
    vb = cinfo_b.get("has_var_embed", False)
    if va and not vb:
        print("\n  🔬  [A] contains the variance head (var_embed) — it is the FINE-TUNED model")
    elif vb and not va:
        print("\n  🔬  [B] contains the variance head (var_embed) — it is the FINE-TUNED model")
    elif va and vb:
        print("\n  🔬  Both checkpoints contain the variance head (var_embed)")
    else:
        print("\n  🔬  Neither checkpoint contains the variance head (var_embed)")

    # Size explanation
    size_a = finfo_a["size_mb"]
    size_b = finfo_b["size_mb"]
    if abs(size_a - size_b) > 10:
        bigger = "A" if size_a > size_b else "B"
        smaller = "B" if size_a > size_b else "A"
        big_mb = max(size_a, size_b)
        small_mb = min(size_a, size_b)
        print(f"\n  ℹ️  Size difference: [{bigger}] {big_mb:.1f} MB  vs  [{smaller}] {small_mb:.1f} MB")
        big_has_opt = cinfo_a.get("has_optimizer") if bigger == "A" else cinfo_b.get("has_optimizer")
        if big_has_opt:
            print(f"       [{bigger}] is larger because it includes optimizer state.")
            print(f"       [{smaller}] contains model weights only — normal for fine-tuned checkpoints.")

    print()


if __name__ == "__main__":
    # Default paths used during Kaggle training runs; will show "File not found" gracefully
    # when run on a different machine.
    _DEFAULT_A = "/kaggle/working/CV_Project/Deformable-DETR/models/checkpoint0000.pth"
    _DEFAULT_B = "/kaggle/working/CV_Project/Deformable-DETR/models/checkpoint.pth"

    if len(sys.argv) == 3:
        ckpt_a = sys.argv[1]
        ckpt_b = sys.argv[2]
    elif len(sys.argv) == 1:
        # Try to discover checkpoints relative to the script directory first.
        _models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Deformable-DETR", "models",
        )
        _local_a = os.path.join(_models_dir, "checkpoint0000.pth")
        _local_b = os.path.join(_models_dir, "checkpoint.pth")
        if os.path.exists(_local_a) or os.path.exists(_local_b):
            ckpt_a, ckpt_b = _local_a, _local_b
        else:
            ckpt_a, ckpt_b = _DEFAULT_A, _DEFAULT_B
        print(f"No paths given — using:\n  A: {ckpt_a}\n  B: {ckpt_b}\n")
    else:
        print(__doc__)
        sys.exit(1)

    compare_checkpoints(ckpt_a, ckpt_b)

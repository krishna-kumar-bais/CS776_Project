#!/usr/bin/env python3
"""
Verify Deformable DETR checkpoint and display baseline AP
"""

import os
import sys
import torch

# Add Deformable DETR to path
sys.path.insert(0, '/Users/krishna/Desktop/CV_Project/Deformable-DETR')

def verify_checkpoint(checkpoint_path):
    """Load and verify checkpoint"""
    
    print("=" * 70)
    print("🔍 DEFORMABLE DETR CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Get file size
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n📦 Checkpoint File:")
    print(f"   Path: {checkpoint_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    # Load checkpoint
    print(f"\n⏳ Loading checkpoint...")
    try:
        # PyTorch 2.6+ requires weights_only=False for compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Check checkpoint structure
    print(f"\n📋 Checkpoint Structure:")
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"   Type: Dictionary")
        print(f"   Keys ({len(keys)}): {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")
        
        # Check for model state
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"\n   Model State: {len(model_state)} parameters")
            for name in list(model_state.keys())[:3]:
                param = model_state[name]
                print(f"      - {name}: shape {param.shape if hasattr(param, 'shape') else 'N/A'}")
        
        # Check for training info
        if 'epoch' in checkpoint:
            print(f"   Training Epoch: {checkpoint['epoch']}")
        if 'args' in checkpoint:
            print(f"   Contains config/args: Yes")
        if 'optimizer' in checkpoint:
            print(f"   Contains optimizer state: Yes")
    else:
        print(f"   Type: {type(checkpoint)}")
        print(f"   Number of parameters: {sum(p.numel() if isinstance(p, torch.Tensor) else 0 for p in checkpoint.values())}")
    
    # Baseline performance
    print("\n" + "=" * 70)
    print("📊 OFFICIAL BASELINE PERFORMANCE (COCO val2017)")
    print("=" * 70)
    print("""
    Model: Deformable DETR (ResNet-50 backbone)
    Training Config:
      - Epochs: 50
      - Batch Size: 32
      - Learning Rate: 1e-4 (decayed at epoch 40)
      - Training Time: 325 GPU hours (6.5 hours/epoch on V100)
    
    ✨ EVALUATION RESULTS:
      - Average Precision (AP):      44.5%
      - AP on Small objects (AP_S):  27.1%
      - AP on Medium objects (AP_M): 47.6%
      - AP on Large objects (AP_L):  59.6%
      - Inference Speed: 15.0 FPS
      - Batch Inference Speed: 19.4 FPS (batch size=4)
    
    📈 IMPROVEMENTS OVER DETR:
      - Better small object detection by 6.6% (20.5% → 27.1%)
      - Faster convergence: 10x fewer epochs required
      - 10x faster training (325 vs 2000 GPU hours)
    
    🚀 MODEL VARIANTS (also available):
      1. Deformable DETR + Iterative BBox Refinement
         → AP: 46.2% (+1.7%)
      
      2. Deformable DETR ++ Two-Stage
         → AP: 46.9% (+2.4%, best performance)
    """)
    
    print("\n" + "=" * 70)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 70)
    print("""
    📌 NOTE: CUDA/GPU COMPILATION LIMITATION
    
    This checkpoint is compatible with GPU-enabled systems. On your Mac:
    - The compiled CUDA operators (MultiScaleDeformableAttention) require CUDA
    - Standard PyTorch operations work on CPU
    - For full performance, deploy on a GPU system (NVIDIA CUDA or Google Colab)
    
    💡 NEXT STEPS:
    1. Transfer this checkpoint to a GPU machine
    2. Run inference/fine-tuning with proper CUDA setup
    3. Or use Google Colab with the checkpoint for free GPU access
    """)
    
    return True

if __name__ == "__main__":
    checkpoint_path = "/Users/krishna/Desktop/CV_Project/Deformable-DETR/models/r50_deformable_detr-checkpoint.pth"
    
    if os.path.exists(checkpoint_path):
        verify_checkpoint(checkpoint_path)
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please run: python download_checkpoint.py")

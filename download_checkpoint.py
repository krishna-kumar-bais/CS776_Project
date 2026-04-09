#!/usr/bin/env python3
"""
Download official Deformable DETR checkpoint from Google Drive
"""

import os
import gdown

# Google Drive links for pre-trained models
MODELS = {
    "deformable_detr": {
        "url": "https://drive.google.com/uc?id=1nDWZWHuRwtwGden77NLM9JoWe-YisJnA",
        "output": "Deformable-DETR/models/r50_deformable_detr-checkpoint.pth",
        "description": "Deformable DETR (50 epochs, AP: 44.5%)"
    },
    "deformable_detr_plus": {
        "url": "https://drive.google.com/uc?id=1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI",
        "output": "Deformable-DETR/models/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth",
        "description": "Deformable DETR + Iterative BBox Refinement (50 epochs, AP: 46.2%)"
    },
    "deformable_detr_pp": {
        "url": "https://drive.google.com/uc?id=15I03A7hNTpwuLNdfuEmW9_taZMNVssEp",
        "output": "Deformable-DETR/models/r50_deformable_detr_two_stage-checkpoint.pth",
        "description": "Deformable DETR ++ Two-Stage (50 epochs, AP: 46.9%)"
    }
}

def download_checkpoint(model_name="deformable_detr"):
    """Download checkpoint from Google Drive"""
    if model_name not in MODELS:
        print(f"Available models: {', '.join(MODELS.keys())}")
        return None
    
    model_info = MODELS[model_name]
    output_path = model_info["output"]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n📥 Downloading: {model_info['description']}")
    print(f"   URL: {model_info['url']}")
    print(f"   Output: {output_path}\n")
    
    try:
        gdown.download(model_info["url"], output_path, quiet=False)
        print(f"✅ Successfully downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deformable_detr"
    checkpoint_path = download_checkpoint(model_name)
    
    if checkpoint_path:
        print(f"\n📦 Checkpoint ready at: {checkpoint_path}")

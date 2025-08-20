#!/usr/bin/env python3
"""
Simple CLIP Test - Download model and basic functionality
"""

import clip
import torch
import time

def simple_clip_test():
    print("🔄 Downloading and loading CLIP model...")
    
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model (will download first time)
    model, preprocess = clip.load("ViT-B/32", device=device)
    load_time = time.time() - start_time
    
    print(f"✅ CLIP model loaded successfully in {load_time:.2f}s")
    
    # Test text embedding
    text_query = "red car"
    print(f"Computing text embedding for: '{text_query}'")
    
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print(f"✅ Text embedding computed: {text_features.shape}")
    print(f"Text embedding sample: {text_features[0][:5].cpu().numpy()}")
    
    print("\n🎉 CLIP setup successful! Ready for image processing.")
    return model, preprocess, device

if __name__ == "__main__":
    model, preprocess, device = simple_clip_test()
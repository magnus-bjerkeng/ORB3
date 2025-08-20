#!/usr/bin/env python3
"""
CLIP Test Script for AGZ Dataset
Tests CLIP functionality with sample drone images and "red car" query
"""

import clip
import torch
import cv2
import numpy as np
import time
import os
from PIL import Image

def test_clip_with_agz_images():
    print("🚀 Testing CLIP with AGZ Dataset Images")
    print("=" * 50)
    
    # Load CLIP model
    print("Loading CLIP model...")
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    load_time = time.time() - start_time
    print(f"✅ CLIP model loaded in {load_time:.2f}s on {device}")
    
    # Define text query
    text_query = "red car"
    print(f"🎯 Text query: '{text_query}'")
    
    # Compute text embedding (pre-compute for efficiency)
    print("Computing text embedding...")
    start_time = time.time()
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_time = time.time() - start_time
    print(f"✅ Text embedding computed in {text_time:.3f}s")
    
    # Find sample AGZ images
    agz_image_dir = "../../AGZ/MAV Images"
    if not os.path.exists(agz_image_dir):
        print(f"❌ AGZ image directory not found: {agz_image_dir}")
        return
    
    # Get first 5 images for testing
    image_files = sorted([f for f in os.listdir(agz_image_dir) if f.endswith('.jpg')])[:5]
    
    if not image_files:
        print("❌ No JPG images found in AGZ directory")
        return
    
    print(f"📸 Testing with {len(image_files)} sample images")
    print("-" * 30)
    
    total_inference_time = 0
    results = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(agz_image_dir, img_file)
        
        try:
            # Load and preprocess image
            image = Image.open(img_path)
            processed_image = preprocess(image).unsqueeze(0).to(device)
            
            # Compute image embedding and similarity
            start_time = time.time()
            with torch.no_grad():
                image_features = model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (text_features @ image_features.T).item()
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            results.append((img_file, similarity, inference_time))
            
            print(f"Image {i+1}: {img_file}")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Time: {inference_time:.3f}s")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    # Performance summary
    print("=" * 50)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Total images processed: {len(results)}")
    print(f"Average inference time: {total_inference_time/len(results):.3f}s per image")
    print(f"Estimated FPS: {len(results)/total_inference_time:.1f}")
    print()
    
    # Find best matches
    if results:
        best_match = max(results, key=lambda x: x[1])
        print(f"🎯 Best match for '{text_query}':")
        print(f"   Image: {best_match[0]}")
        print(f"   Similarity: {best_match[1]:.4f}")
        print()
        
        # Similarity thresholds for visualization
        print("🎨 Suggested visualization thresholds:")
        all_similarities = [r[1] for r in results]
        max_sim = max(all_similarities)
        min_sim = min(all_similarities)
        print(f"   Low (Green):    < {min_sim + 0.1:.3f}")
        print(f"   Medium (Yellow): {min_sim + 0.1:.3f} - {max_sim - 0.1:.3f}")
        print(f"   High (Red):     > {max_sim - 0.1:.3f}")
        print()
    
    # Frame sampling recommendations
    print("⚡ Real-time processing recommendations:")
    avg_time = total_inference_time / len(results) if results else 0
    if avg_time > 0:
        target_fps = 30  # SLAM processing rate
        clip_fps = 1 / avg_time
        sampling_interval = max(1, int(target_fps / min(5, clip_fps)))  # Max 5 FPS for CLIP
        print(f"   Process every {sampling_interval} frames for real-time performance")
        print(f"   This gives ~{target_fps/sampling_interval:.1f} FPS CLIP analysis")
    
    return results

def benchmark_different_queries():
    """Test different text queries to understand performance"""
    print("\n🔬 BENCHMARKING DIFFERENT QUERIES")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    queries = [
        "red car",
        "building",
        "street",
        "person walking",
        "green tree",
        "blue sky"
    ]
    
    print("Computing text embeddings for multiple queries...")
    start_time = time.time()
    text = clip.tokenize(queries).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    multi_text_time = time.time() - start_time
    
    print(f"✅ {len(queries)} text embeddings computed in {multi_text_time:.3f}s")
    print(f"   Average: {multi_text_time/len(queries):.4f}s per query")
    print("   💡 Text embeddings can be pre-computed and cached!")

if __name__ == "__main__":
    # Test basic CLIP functionality
    results = test_clip_with_agz_images()
    
    # Benchmark multiple queries
    benchmark_different_queries()
    
    print("\n🎉 CLIP testing complete! Ready for integration with ORB-SLAM3.")
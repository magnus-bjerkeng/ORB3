#!/usr/bin/env python3
"""
CLIP Service for Real-time Semantic Analysis
Provides text-image similarity scoring for ORB-SLAM3 visualization
"""

import clip
import torch
import cv2
import numpy as np
import time
import os
import sys
import json
import threading
import queue
import mmap
from PIL import Image
import argparse
import logging

class CLIPService:
    def __init__(self, device="cpu", model_name="ViT-B/32"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.text_features_cache = {}
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.start_time = time.time()
        
        # Configuration
        self.similarity_thresholds = {
            "low": 0.15,      # Green drone
            "medium": 0.25,   # Yellow drone  
            "high": 0.35      # Red drone
        }
        
        # Communication queues
        self.image_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory buildup
        self.result_queue = queue.Queue()
        
        # Current query
        self.current_query = "red car"
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the service"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - CLIP Service - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('clip_service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self):
        """Load and initialize CLIP model"""
        self.logger.info(f"Loading CLIP model {self.model_name} on {self.device}...")
        start_time = time.time()
        
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        load_time = time.time() - start_time
        self.logger.info(f"✅ CLIP model loaded in {load_time:.2f}s")
        return True
    
    def update_text_query(self, query):
        """Update the text query and compute embedding"""
        if query == self.current_query and query in self.text_features_cache:
            return  # No change needed
            
        self.logger.info(f"Updating text query to: '{query}'")
        self.current_query = query
        
        # Compute and cache text embedding
        start_time = time.time()
        text = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        self.text_features_cache[query] = text_features
        compute_time = time.time() - start_time
        self.logger.info(f"Text embedding computed in {compute_time:.3f}s")
    
    def process_image_array(self, image_array):
        """Process numpy image array and return similarity score"""
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess for CLIP
            processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get cached text features
            text_features = self.text_features_cache.get(self.current_query)
            if text_features is None:
                self.update_text_query(self.current_query)
                text_features = self.text_features_cache[self.current_query]
            
            # Compute similarity
            start_time = time.time()
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (text_features @ image_features.T).item()
            
            inference_time = time.time() - start_time
            
            # Update performance tracking
            self.frame_count += 1
            self.total_inference_time += inference_time
            
            # Determine color based on similarity thresholds
            if similarity >= self.similarity_thresholds["high"]:
                color_code = "red"
            elif similarity >= self.similarity_thresholds["medium"]:
                color_code = "yellow"
            elif similarity >= self.similarity_thresholds["low"]:
                color_code = "orange"
            else:
                color_code = "green"
            
            return {
                "similarity": similarity,
                "color_code": color_code,
                "query": self.current_query,
                "inference_time": inference_time,
                "frame_id": self.frame_count
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {
                "similarity": 0.0,
                "color_code": "green",
                "query": self.current_query,
                "inference_time": 0.0,
                "frame_id": self.frame_count,
                "error": str(e)
            }
    
    def process_image_file(self, image_path):
        """Process image file and return similarity score"""
        try:
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Could not load image: {image_path}")
            return self.process_image_array(image_array)
        except Exception as e:
            self.logger.error(f"Error processing image file {image_path}: {e}")
            return {"similarity": 0.0, "color_code": "green", "error": str(e)}
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        runtime = time.time() - self.start_time
        avg_inference_time = self.total_inference_time / max(1, self.frame_count)
        fps = self.frame_count / max(0.001, self.total_inference_time)
        
        return {
            "frames_processed": self.frame_count,
            "runtime_seconds": runtime,
            "avg_inference_time": avg_inference_time,
            "estimated_fps": fps,
            "current_query": self.current_query,
            "device": self.device
        }
    
    def print_performance_stats(self):
        """Print performance statistics"""
        stats = self.get_performance_stats()
        self.logger.info("=" * 50)
        self.logger.info("📊 CLIP SERVICE PERFORMANCE")
        self.logger.info("=" * 50)
        self.logger.info(f"Frames processed: {stats['frames_processed']}")
        self.logger.info(f"Runtime: {stats['runtime_seconds']:.1f}s")
        self.logger.info(f"Average inference: {stats['avg_inference_time']:.3f}s")
        self.logger.info(f"Estimated FPS: {stats['estimated_fps']:.1f}")
        self.logger.info(f"Current query: '{stats['current_query']}'")
        self.logger.info(f"Device: {stats['device']}")
    
    def start_interactive_mode(self):
        """Start interactive mode for testing"""
        self.logger.info("🚀 CLIP Service Interactive Mode Started")
        self.logger.info("Commands: 'query <text>', 'test <image_path>', 'stats', 'quit'")
        
        if not self.initialize_model():
            return
        
        # Initialize with default query
        self.update_text_query(self.current_query)
        
        while True:
            try:
                command = input("\nCLIP> ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'stats':
                    self.print_performance_stats()
                elif command.startswith('query '):
                    new_query = command[6:].strip()
                    if new_query:
                        self.update_text_query(new_query)
                    else:
                        self.logger.info(f"Current query: '{self.current_query}'")
                elif command.startswith('test '):
                    image_path = command[5:].strip()
                    if os.path.exists(image_path):
                        result = self.process_image_file(image_path)
                        self.logger.info(f"Image: {os.path.basename(image_path)}")
                        self.logger.info(f"Similarity: {result['similarity']:.4f}")
                        self.logger.info(f"Color: {result['color_code']}")
                        self.logger.info(f"Time: {result['inference_time']:.3f}s")
                    else:
                        self.logger.error(f"Image not found: {image_path}")
                elif command == '':
                    continue
                else:
                    self.logger.info("Unknown command. Use 'query <text>', 'test <image>', 'stats', or 'quit'")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
        
        self.print_performance_stats()
        self.logger.info("CLIP Service stopped.")

def main():
    parser = argparse.ArgumentParser(description='CLIP Service for ORB-SLAM3 Semantic Analysis')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to run CLIP on (default: cpu)')
    parser.add_argument('--model', default='ViT-B/32', 
                       help='CLIP model to use (default: ViT-B/32)')
    parser.add_argument('--query', default='red car',
                       help='Initial text query (default: red car)')
    parser.add_argument('--interactive', action='store_true',
                       help='Start in interactive mode')
    parser.add_argument('--test-image', 
                       help='Test with a single image file')
    
    args = parser.parse_args()
    
    # Initialize CLIP service
    service = CLIPService(device=args.device, model_name=args.model)
    
    if not service.initialize_model():
        return 1
    
    # Set initial query
    service.update_text_query(args.query)
    
    if args.interactive:
        service.start_interactive_mode()
    elif args.test_image:
        if os.path.exists(args.test_image):
            result = service.process_image_file(args.test_image)
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: Image file not found: {args.test_image}")
            return 1
    else:
        print("Use --interactive for interactive mode or --test-image <path> to test a single image")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
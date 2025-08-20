#!/usr/bin/env python3
"""
CLIP-SLAM Integration Service
File-based communication service for real-time SLAM-CLIP integration
Monitors shared directory for image processing requests from mono_agz_clip
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
import argparse
import logging
from PIL import Image
from pathlib import Path

class CLIPSLAMService:
    def __init__(self, shared_dir, device="cpu", model_name="ViT-B/32", query="red car"):
        self.shared_dir = Path(shared_dir)
        self.device = device
        self.model_name = model_name
        self.current_query = query
        
        # Initialize paths
        self.images_dir = self.shared_dir / "images"
        self.results_dir = self.shared_dir / "results"
        self.config_path = self.shared_dir / "config.txt"
        self.query_update_path = self.shared_dir / "query_update.txt"
        
        # CLIP model
        self.model = None
        self.preprocess = None
        self.text_features_cache = {}
        
        # Performance tracking
        self.processed_frames = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        
        # Configuration defaults
        self.similarity_thresholds = {
            "low": 0.15,      # Green
            "medium": 0.25,   # Orange/Yellow  
            "high": 0.35      # Red
        }
        
        # Service control
        self.running = False
        
        self.setup_logging()
        self.create_directories()
    
    def setup_logging(self):
        """Setup logging for the service"""
        log_file = self.shared_dir / "clip_slam_service.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - CLIP-SLAM Service - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories"""
        self.shared_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created directories: {self.shared_dir}")
    
    def initialize_model(self):
        """Load and initialize CLIP model"""
        self.logger.info(f"Loading CLIP model {self.model_name} on {self.device}...")
        start_time = time.time()
        
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            load_time = time.time() - start_time
            self.logger.info(f"✅ CLIP model loaded in {load_time:.2f}s")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            return False
    
    def load_config(self):
        """Load configuration from shared directory"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if key == 'query':
                                self.current_query = value
                            elif key.startswith('thresholds_'):
                                threshold_type = key.split('_')[1]
                                self.similarity_thresholds[threshold_type] = float(value)
                
                self.logger.info(f"Configuration loaded: query='{self.current_query}'")
                self.logger.info(f"Thresholds: {self.similarity_thresholds}")
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
    
    def update_text_query(self, query):
        """Update the text query and compute embedding"""
        if query == self.current_query and query in self.text_features_cache:
            return
            
        self.logger.info(f"Updating text query to: '{query}'")
        self.current_query = query
        
        try:
            # Compute and cache text embedding
            start_time = time.time()
            text = clip.tokenize([query]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            self.text_features_cache[query] = text_features
            compute_time = time.time() - start_time
            self.logger.info(f"Text embedding computed in {compute_time:.3f}s")
        except Exception as e:
            self.logger.error(f"Failed to update text query: {e}")
    
    def process_image_request(self, request_file):
        """Process a single image request from SLAM system"""
        try:
            # Parse request file
            request_data = {}
            with open(request_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        request_data[key] = value
            
            frame_id = int(request_data.get('frame_id', 0))
            image_path = request_data.get('image_path', '')
            
            # Load and process image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess for CLIP
            processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get text features
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
            
            # Determine color based on similarity
            if similarity >= self.similarity_thresholds["high"]:
                color_code = "red"
            elif similarity >= self.similarity_thresholds["medium"]:
                color_code = "orange"  # Changed from yellow to orange
            elif similarity >= self.similarity_thresholds["low"]:
                color_code = "yellow"  # Light activity
            else:
                color_code = "green"
            
            # Create result
            result = {
                "similarity": similarity,
                "color_code": color_code,
                "query": self.current_query,
                "inference_time": inference_time,
                "frame_id": frame_id,
                "timestamp": time.time()
            }
            
            # Save result
            result_file = self.results_dir / f"frame_{frame_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Update statistics
            self.processed_frames += 1
            self.total_inference_time += inference_time
            
            # Log high similarity detections
            if similarity >= self.similarity_thresholds["medium"]:
                self.logger.info(f"Frame {frame_id}: '{self.current_query}' = {similarity:.3f} ({color_code})")
            
            # Clean up request and image files
            os.remove(request_file)
            os.remove(image_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request {request_file}: {e}")
            # Still clean up files
            try:
                os.remove(request_file)
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
            return None
    
    def check_query_updates(self):
        """Check for query updates from SLAM system"""
        if self.query_update_path.exists():
            try:
                with open(self.query_update_path, 'r') as f:
                    new_query = f.read().strip()
                if new_query and new_query != self.current_query:
                    self.update_text_query(new_query)
                os.remove(self.query_update_path)
            except Exception as e:
                self.logger.error(f"Error checking query updates: {e}")
    
    def run_service(self):
        """Main service loop"""
        self.logger.info("🚀 CLIP-SLAM Integration Service Started")
        self.logger.info(f"Monitoring directory: {self.shared_dir}")
        self.logger.info(f"Initial query: '{self.current_query}'")
        
        if not self.initialize_model():
            return False
        
        # Load configuration
        self.load_config()
        
        # Initialize with current query
        self.update_text_query(self.current_query)
        
        self.running = True
        
        while self.running:
            try:
                # Check for query updates
                self.check_query_updates()
                
                # Process pending requests
                request_files = list(self.shared_dir.glob("*.request"))
                
                if request_files:
                    # Sort by creation time to maintain frame order
                    request_files.sort(key=lambda x: x.stat().st_mtime)
                    
                    for request_file in request_files:
                        if not self.running:
                            break
                        self.process_image_request(request_file)
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.01)  # 10ms polling interval
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Service error: {e}")
                time.sleep(1)
        
        self.stop_service()
    
    def stop_service(self):
        """Stop the service and print statistics"""
        self.running = False
        
        runtime = time.time() - self.start_time
        
        self.logger.info("=" * 50)
        self.logger.info("📊 CLIP-SLAM SERVICE SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Runtime: {runtime:.1f}s")
        self.logger.info(f"Frames processed: {self.processed_frames}")
        if self.processed_frames > 0:
            self.logger.info(f"Average inference time: {(self.total_inference_time / self.processed_frames):.3f}s")
            self.logger.info(f"Processing FPS: {(self.processed_frames / self.total_inference_time):.1f}")
        self.logger.info(f"Final query: '{self.current_query}'")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("CLIP-SLAM Integration Service stopped.")

def main():
    parser = argparse.ArgumentParser(description='CLIP-SLAM Integration Service')
    parser.add_argument('--shared-dir', default='./clip_shared',
                       help='Shared directory for SLAM-CLIP communication')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to run CLIP on (default: cpu)')
    parser.add_argument('--model', default='ViT-B/32', 
                       help='CLIP model to use (default: ViT-B/32)')
    parser.add_argument('--query', default='red car',
                       help='Initial text query (default: red car)')
    
    args = parser.parse_args()
    
    # Initialize and run service
    service = CLIPSLAMService(
        shared_dir=args.shared_dir,
        device=args.device,
        model_name=args.model,
        query=args.query
    )
    
    try:
        service.run_service()
        return 0
    except Exception as e:
        service.logger.error(f"Service failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
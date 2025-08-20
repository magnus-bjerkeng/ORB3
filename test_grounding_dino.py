#!/usr/bin/env python3
"""
Grounding DINO Test Script for Frame 5904
=========================================

Tests open vocabulary object detection on frame 5904 with "red car" query.
Generates bounding box detections with confidence scores and visualization.

Usage:
    python3 test_grounding_dino.py

Requirements:
    - Grounding DINO installed
    - Model weights in grounding_dino_workspace/GroundingDINO/weights/
    - frame_5904_test.jpg in current directory
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time

# Add Grounding DINO to path
sys.path.insert(0, 'grounding_dino_workspace/GroundingDINO')

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
except ImportError as e:
    print(f"Error importing Grounding DINO: {e}")
    print("Make sure Grounding DINO is properly installed and in the correct path")
    sys.exit(1)


class GroundingDINOTester:
    """Grounding DINO tester for frame 5904 red car detection."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Paths
        self.model_config_path = "grounding_dino_workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.model_checkpoint_path = "grounding_dino_workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.test_image_path = "frame_5904_test.jpg"
        self.results_dir = "results"
        
        # Detection parameters (optimized configuration)
        self.text_prompt = "red car"
        self.box_threshold = 0.25  # Optimized: was 0.35, now 0.25 for more detections
        self.text_threshold = 0.15  # Optimized: was 0.25, now 0.15 for broader matching
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_model(self):
        """Load Grounding DINO model with weights."""
        print("Loading Grounding DINO model...")
        
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model config not found: {self.model_config_path}")
        
        if not os.path.exists(self.model_checkpoint_path):
            raise FileNotFoundError(f"Model weights not found: {self.model_checkpoint_path}")
        
        try:
            # Load model using the utility function with device specification
            device_str = "cpu" if self.device.type == "cpu" else "cuda"
            self.model = load_model(self.model_config_path, self.model_checkpoint_path, device=device_str)
            print("✓ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def run_inference(self):
        """Run inference on frame 5904 with 'red car' query."""
        print(f"\nRunning inference on {self.test_image_path}...")
        print(f"Query: '{self.text_prompt}'")
        print(f"Box threshold: {self.box_threshold}")
        print(f"Text threshold: {self.text_threshold}")
        
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Test image not found: {self.test_image_path}")
        
        # Load and preprocess image
        image_source, image = load_image(self.test_image_path)
        
        # Record inference time
        start_time = time.time()
        
        # Run prediction (with device specification for CPU)
        device_str = "cpu" if self.device.type == "cpu" else "cuda"
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=device_str
        )
        
        inference_time = time.time() - start_time
        
        # Print results
        print(f"\n=== DETECTION RESULTS ===")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Detected objects: {len(boxes)}")
        
        if len(boxes) > 0:
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                confidence = logit.item()
                x1, y1, x2, y2 = box.cpu().numpy()
                print(f"Detection {i+1}:")
                print(f"  Phrase: {phrase}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Bounding box: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
        else:
            print("No objects detected!")
        
        return image_source, boxes, logits, phrases, inference_time
    
    def visualize_results(self, image_source, boxes, logits, phrases, inference_time):
        """Create visualization with bounding boxes and save results."""
        print(f"\nGenerating visualization...")
        
        # Create annotated image using Grounding DINO's utility
        # annotate() works with RGB input but internally swaps R and B channels
        annotated_frame_swapped = annotate(
            image_source=image_source, 
            boxes=boxes, 
            logits=logits, 
            phrases=phrases
        )
        # Fix the color channel swap (R and B are swapped by annotate)
        annotated_frame = annotated_frame_swapped[:, :, [2, 1, 0]]  # Swap R and B back
        
        # Save annotated image (convert to BGR for OpenCV saving)
        output_path = os.path.join(self.results_dir, "frame_5904_grounding_dino_results.jpg")
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_frame_bgr)
        print(f"✓ Saved annotated image: {output_path}")
        
        # Create detailed matplotlib visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image (image_source from load_image is already RGB)
        ax1.imshow(image_source)
        ax1.set_title("Original Frame 5904", fontsize=16)
        ax1.axis('off')
        
        # Annotated image with results (annotated_frame is also RGB)
        ax2.imshow(annotated_frame)
        ax2.set_title(f"Grounding DINO Results: '{self.text_prompt}'\n"
                     f"Detections: {len(boxes)} | Inference: {inference_time:.2f}s", fontsize=16)
        ax2.axis('off')
        
        # Add detection details as text
        if len(boxes) > 0:
            details = []
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                confidence = logit.item()
                details.append(f"Detection {i+1}: {phrase} ({confidence:.3f})")
            
            details_text = "\n".join(details)
            ax2.text(0.02, 0.98, details_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round", 
                    facecolor='white', alpha=0.8), fontsize=12)
        
        plt.tight_layout()
        
        # Save detailed visualization
        detailed_output_path = os.path.join(self.results_dir, "frame_5904_detailed_results.png")
        plt.savefig(detailed_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved detailed visualization: {detailed_output_path}")
        
        return output_path, detailed_output_path
    
    def save_detection_data(self, boxes, logits, phrases, inference_time):
        """Save detection data to text files for further analysis."""
        # Save raw detection data
        data_file = os.path.join(self.results_dir, "detection_data.txt")
        with open(data_file, 'w') as f:
            f.write(f"Grounding DINO Detection Results\n")
            f.write(f"================================\n")
            f.write(f"Image: frame_5904_test.jpg\n")
            f.write(f"Query: {self.text_prompt}\n")
            f.write(f"Box threshold: {self.box_threshold}\n")
            f.write(f"Text threshold: {self.text_threshold}\n")
            f.write(f"Inference time: {inference_time:.3f} seconds\n")
            f.write(f"Total detections: {len(boxes)}\n\n")
            
            if len(boxes) > 0:
                for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                    confidence = logit.item()
                    x1, y1, x2, y2 = box.cpu().numpy()
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Phrase: {phrase}\n")
                    f.write(f"  Confidence: {confidence:.6f}\n")
                    f.write(f"  Bounding box (normalized): [{x1:.6f}, {y1:.6f}, {x2:.6f}, {y2:.6f}]\n\n")
            else:
                f.write("No detections found.\n")
        
        print(f"✓ Saved detection data: {data_file}")
        return data_file
    
    def run_full_test(self):
        """Run complete Grounding DINO test pipeline."""
        print("="*60)
        print("🚗 GROUNDING DINO - FRAME 5904 RED CAR DETECTION TEST")
        print("="*60)
        
        try:
            # Load model
            if not self.load_model():
                return False
            
            # Run inference
            image_source, boxes, logits, phrases, inference_time = self.run_inference()
            
            # Create visualizations
            output_path, detailed_path = self.visualize_results(
                image_source, boxes, logits, phrases, inference_time
            )
            
            # Save detection data
            data_file = self.save_detection_data(boxes, logits, phrases, inference_time)
            
            print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved in: {self.results_dir}/")
            print(f"📸 Annotated image: {output_path}")
            print(f"📊 Detailed visualization: {detailed_path}")
            print(f"📄 Detection data: {data_file}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the Grounding DINO test."""
    tester = GroundingDINOTester()
    success = tester.run_full_test()
    
    if success:
        print(f"\n🎉 Red car detection test completed successfully!")
    else:
        print(f"\n💥 Red car detection test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
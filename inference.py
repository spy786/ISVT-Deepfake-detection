import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import List, Tuple

from config import config
from model import ISTVT
from utils import FaceDetector, load_checkpoint
import torchvision.transforms as transforms

class ISVTInference:
    """Inference and visualization for ISTVT"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ISTVT(config).to(self.device)
        load_checkpoint(model_path, self.model)
        self.model.eval()
        
        # Face detector and transforms
        self.face_detector = FaceDetector()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_video(self, video_path: str) -> Tuple[float, str]:
        """Predict if video is fake or real"""
        
        # Extract face sequence
        face_sequence = self.face_detector.extract_sequence(video_path, config.sequence_length)
        
        if len(face_sequence) < config.sequence_length:
            return 0.0, "Could not extract enough faces"
        
        # Prepare input
        video_frames = []
        for face in face_sequence:
            frame_tensor = self.transform(face)
            video_frames.append(frame_tensor)
        
        video_tensor = torch.stack(video_frames).unsqueeze(0)  # (1, T, C, H, W)
        video_tensor = video_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(video_tensor)
            fake_prob = torch.sigmoid(outputs.squeeze()).item()
        
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        
        return fake_prob, prediction
    
    def visualize_sequence(self, video_path: str, save_path: str = None):
        """Visualize face sequence from video"""
        
        face_sequence = self.face_detector.extract_sequence(video_path, config.sequence_length)
        
        if len(face_sequence) == 0:
            print("No faces detected in video")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, face in enumerate(face_sequence[:6]):
            if i < len(axes):
                axes[i].imshow(face)
                axes[i].set_title(f'Frame {i+1}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(face_sequence), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def batch_predict(self, video_dir: str) -> List[Tuple[str, float, str]]:
        """Predict on batch of videos"""
        
        results = []
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            
            try:
                fake_prob, prediction = self.predict_video(video_path)
                results.append((video_file, fake_prob, prediction))
                print(f"{video_file}: {prediction} (confidence: {fake_prob:.3f})")
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                results.append((video_file, 0.0, "ERROR"))
        
        return results
    
    def create_detection_report(self, results: List[Tuple[str, float, str]], save_path: str = "detection_report.txt"):
        """Create detection report"""
        
        with open(save_path, 'w') as f:
            f.write("ISTVT Deepfake Detection Report\n")
            f.write("=" * 50 + "\n\n")
            
            fake_count = sum(1 for _, _, pred in results if pred == "FAKE")
            real_count = sum(1 for _, _, pred in results if pred == "REAL")
            error_count = sum(1 for _, _, pred in results if pred == "ERROR")
            
            f.write(f"Total videos processed: {len(results)}\n")
            f.write(f"Detected as FAKE: {fake_count}\n")
            f.write(f"Detected as REAL: {real_count}\n")
            f.write(f"Processing errors: {error_count}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for filename, prob, pred in results:
                f.write(f"{filename}: {pred} (confidence: {prob:.3f})\n")
        
        print(f"Detection report saved to {save_path}")

def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ISTVT Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--video_dir', type=str, help='Path to directory containing videos')
    parser.add_argument('--visualize', action='store_true', help='Visualize face sequence')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = ISVTInference(args.model)
    
    if args.video:
        # Single video prediction
        print(f"Processing video: {args.video}")
        fake_prob, prediction = inference.predict_video(args.video)
        print(f"Prediction: {prediction} (confidence: {fake_prob:.3f})")
        
        if args.visualize:
            vis_path = os.path.join(args.output_dir, f"visualization_{os.path.basename(args.video)}.png")
            inference.visualize_sequence(args.video, vis_path)
    
    elif args.video_dir:
        # Batch prediction
        print(f"Processing videos in directory: {args.video_dir}")
        results = inference.batch_predict(args.video_dir)
        
        # Create report
        report_path = os.path.join(args.output_dir, "detection_report.txt")
        inference.create_detection_report(results, report_path)
    
    else:
        print("Please provide either --video or --video_dir argument")

if __name__ == '__main__':
    main()
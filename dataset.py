import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import random

from utils import FaceDetector

class DeepfakeDataset(Dataset):
    """Dataset for deepfake video detection"""
    
    def __init__(self, data_root: str, split: str = 'train', config=None):
        self.data_root = data_root
        self.split = split
        self.config = config
        self.sequence_length = config.sequence_length if config else 6
        
        # Initialize face detector
        device = config.device if config else None
        self.face_detector = FaceDetector(device=device)
        
        # Load video paths and labels
        self.samples = self._load_samples()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self) -> List[Dict]:
        """Load dataset samples"""
        samples = []
        
        # Look for metadata file
        metadata_path = os.path.join(self.data_root, f"{self.split}.json")
        
        if os.path.exists(metadata_path):
            # Load from metadata file
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for item in metadata:
                samples.append({
                    'video_path': os.path.join(self.data_root, item['path']),
                    'label': item['label']
                })
        else:
            # Create simple structure: real/fake folders
            for label_name in ['real', 'fake']:
                label_dir = os.path.join(self.data_root, self.split, label_name)
                if os.path.exists(label_dir):
                    label = 0 if label_name == 'real' else 1
                    
                    for video_file in os.listdir(label_dir):
                        if video_file.endswith(('.mp4', '.avi', '.mov')):
                            samples.append({
                                'video_path': os.path.join(label_dir, video_file),
                                'label': label
                            })
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        
        # Extract face sequence
        face_sequence = self.face_detector.extract_sequence(video_path, self.sequence_length)
        
        # Handle insufficient faces
        if len(face_sequence) < self.sequence_length:
            if len(face_sequence) == 0:
                # Create dummy sequence
                face_sequence = [np.zeros((300, 300, 3), dtype=np.uint8)] * self.sequence_length
            else:
                # Repeat last frame
                while len(face_sequence) < self.sequence_length:
                    face_sequence.append(face_sequence[-1])
        
        # Apply transforms with temporal consistency
        video_frames = []
        for face in face_sequence:
            frame_tensor = self.transform(face)
            video_frames.append(frame_tensor)
        
        # Stack to create video tensor: (T, C, H, W)
        video_tensor = torch.stack(video_frames)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return video_tensor, label_tensor

def create_dataloaders(config):
    """Create train/val/test dataloaders"""
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = DeepfakeDataset(
            data_root=config.data_root,
            split=split,
            config=config
        )
        datasets[split] = dataset
        
        # Create dataloader
        shuffle = (split == 'train')
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        dataloaders[split] = dataloader
    
    return dataloaders

# Utility function to create sample dataset structure
def create_sample_dataset(data_root: str):
    """Create sample dataset structure for testing"""
    splits = ['train', 'val', 'test']
    labels = ['real', 'fake']
    
    for split in splits:
        for label in labels:
            dir_path = os.path.join(data_root, split, label)
            os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created sample dataset structure in {data_root}")
    print("Place your video files in the appropriate folders:")
    for split in splits:
        for label in labels:
            print(f"  {data_root}/{split}/{label}/")
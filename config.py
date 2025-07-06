import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    """Complete configuration for ISTVT"""
    
    # Model parameters
    sequence_length: int = 6
    input_size: Tuple[int, int] = (300, 300)
    num_channels: int = 3
    embed_dim: int = 728
    num_heads: int = 8
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 1
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Data parameters
    data_root: str = "./data"
    face_margin: float = 1.25
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)

# Global config instance
config = Config()
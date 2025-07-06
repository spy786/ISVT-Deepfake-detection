# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

**Course Project Implementation for EE656**

This repository contains a PyTorch implementation of the ISTVT (Interpretable Spatial-Temporal Video Transformer) model for deepfake detection, developed as a course project for EE656. The implementation is based on the paper "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection" but represents an independent implementation for educational purposes.

## Project Overview

This implementation focuses on video-based deepfake detection using a transformer architecture that processes spatial and temporal information separately to achieve computational efficiency while maintaining detection accuracy.

### Key Features

- **Decomposed Spatial-Temporal Attention**: Processes spatial and temporal dimensions separately to reduce computational complexity
- **Self-Subtract Mechanism**: Captures temporal inconsistencies between consecutive frames
- **Face-Centered Processing**: Uses MTCNN for face detection and alignment
- **Configurable Architecture**: Easily adjustable model parameters through configuration file

## Project Structure

```
ISTVT_deepfake_det/
├── config.py          # Model and training configuration
├── dataset.py         # Dataset handling and data loading
├── inference.py       # Inference and prediction functionality  
├── model.py           # ISTVT model implementation
├── train.py           # Training script
├── utils.py           # Utility functions (face detection, metrics, etc.)
├── requirements.txt   # Python dependencies
└── data/             # Dataset directory (to be created)
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/saubhagyapandey27/ISTVT_deepfake_det/
cd ee656_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create data directory structure**
```bash
mkdir -p data/{train,val,test}/{real,fake}
```

## Dataset Preparation

Organize your video dataset in the following structure:

```
data/
├── train/
│   ├── real/     # Real videos for training
│   └── fake/     # Fake videos for training
├── val/
│   ├── real/     # Real videos for validation
│   └── fake/     # Fake videos for validation
└── test/
    ├── real/     # Real videos for testing
    └── fake/     # Fake videos for testing
```

**Supported video formats**: `.mp4`, `.avi`, `.mov`

## Model Architecture

### Core Components

1. **Feature Extractor**: Xception entry flow (first 3 blocks) for texture feature extraction
2. **Classification Tokens**: Three types of tokens for different purposes:
   - Prediction CLS token (final classification)
   - Temporal CLS token (temporal aggregation)
   - Spatial CLS tokens (spatial features for each frame)
3. **Decomposed Attention**: Separate spatial and temporal self-attention mechanisms
4. **Self-Subtract Mechanism**: Temporal attention operates on frame differences
5. **Transformer Blocks**: 12 layers with 8-head attention

### Model Parameters

- **Input**: 6 consecutive frames of 300×300 RGB images
- **Feature Dimension**: 728
- **Attention Heads**: 8
- **Transformer Layers**: 12
- **Output**: Binary classification (real/fake)

## Usage

### Training

```bash
python train.py
```

**Training Features**:
- Automatic checkpoint saving
- Tensorboard logging
- Gradient clipping
- Cosine annealing scheduler
- Warmup epochs

**Outputs**:
- Model checkpoints: `./checkpoints/`
- Training logs: `./logs/`
- Tensorboard logs: `./logs/`

### Inference

**Single video prediction**:
```bash
python inference.py --model checkpoints/best_model.pth --video path/to/video.mp4
```

**Batch processing**:
```bash
python inference.py --model checkpoints/best_model.pth --video_dir path/to/videos/ --output_dir ./results
```

**With visualization**:
```bash
python inference.py --model checkpoints/best_model.pth --video path/to/video.mp4 --visualize
```

### Configuration

Modify `config.py` to adjust model and training parameters:

```python
# Model parameters
sequence_length: int = 6        # Number of frames per video
input_size: Tuple[int, int] = (300, 300)  # Input image size
embed_dim: int = 728           # Feature embedding dimension
num_heads: int = 8             # Number of attention heads
num_layers: int = 12           # Number of transformer blocks

# Training parameters  
batch_size: int = 4            # Batch size
learning_rate: float = 0.0005  # Learning rate
num_epochs: int = 100          # Training epochs
```

## Implementation Details

### Face Detection and Preprocessing

- **Face Detection**: MTCNN with confidence thresholds [0.6, 0.7, 0.7]
- **Face Alignment**: Similarity transformation based on eye landmarks
- **Cropping**: Nose-centered cropping with 1.25× margin
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Attention Mechanism

The decomposed attention processes spatial and temporal dimensions separately:

1. **Temporal Attention**: Applied to frame differences (self-subtract mechanism)
2. **Spatial Attention**: Applied to spatial tokens within each frame
3. **Computational Complexity**: Reduced from O(T²H²W²) to O(T² + H²W²)

### Training Strategy

- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: SGD with momentum (0.9) and weight decay (1e-4)
- **Scheduler**: Cosine annealing with warmup
- **Gradient Clipping**: Max norm of 1.0

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **AUC**: Area under ROC curve

## Limitations and Future Work

This implementation focuses on the core ISTVT architecture and training pipeline. The following components from the original paper are not implemented:

- **LRP-based Interpretability**: Layer-wise relevance propagation for attention visualization
- **Robustness Testing**: Evaluation under compression, downscaling, and noise
- **Advanced Augmentation**: Specific augmentation strategies for deepfake detection
- **Multi-dataset Evaluation**: Cross-dataset generalization experiments

## Course Project Notes

This implementation was developed as part of the EE656 course curriculum. While it implements the core concepts from the ISTVT paper, it represents an independent educational implementation rather than a reproduction of the original authors' work.

## References

The implementation is based on concepts from:
```
Zhao, Cairong, et al. "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection." 
IEEE Transactions on Information Forensics and Security 18 (2023): 1335-1348.
```

## License

This project is developed for educational purposes as part of the EE656 course.

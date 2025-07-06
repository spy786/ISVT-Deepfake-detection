import cv2
import numpy as np
import torch
import os
from facenet_pytorch import MTCNN
from typing import List, Optional, Tuple
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class FaceDetector:
    """Face detection and preprocessing using MTCNN"""
    
    def __init__(self, device=None, margin=1.25, min_face_size=20):
        # Use specified device
        self.detector = MTCNN(
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            device='cpu'
        )
        self.margin = margin
    
    def align_face(self, frame: np.ndarray, landmarks: dict) -> Optional[np.ndarray]:
        """
        Apply face alignment using similarity transformation based on landmarks
        This ensures consistent face orientation across frames (paper requirement)
        """
        try:
            # Extract key landmarks
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')
            nose = landmarks.get('nose')
            
            if not all([left_eye, right_eye, nose]):
                return None
            
            # Calculate eye center and angle
            eye_center = ((left_eye[0] + right_eye[0]) // 2, 
                        (left_eye[1] + right_eye[1]) // 2)

            # Calculate rotation angle to align eyes horizontally
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Create rotation matrix around eye center (correct approach)
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            
            # Apply rotation
            aligned_frame = cv2.warpAffine(frame, rotation_matrix, 
                                        (frame.shape[1], frame.shape[0]))
            
            return aligned_frame
            
        except Exception as e:
            print(f"Face alignment error: {e}")
            return frame  # Return original if alignment fails

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect, align, and crop face from frame using nose-tip alignment"""
        try:
            boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return None

            # Get largest face (first detection is usually the most confident)
            box = boxes[0]
            x, y, w, h = box.astype(int).tolist()
            
            keypoints = {}
            if landmarks is not None and len(landmarks) > 0:
                kp = landmarks[0]
                keypoints = {
                    'left_eye': kp[0].astype(int).tolist(),
                    'right_eye': kp[1].astype(int).tolist(),
                    'nose': kp[2].astype(int).tolist(),
                    'mouth_left': kp[3].astype(int).tolist(),
                    'mouth_right': kp[4].astype(int).tolist()
                }

            # Apply face alignment first (maintains temporal stability)
            aligned_frame = self.align_face(frame, keypoints)
            if aligned_frame is None:
                aligned_frame = frame

            # Re-detect face in aligned frame for accurate cropping
            aligned_boxes, aligned_probs, aligned_landmarks = self.detector.detect(aligned_frame, landmarks=True)
            
            if aligned_boxes is not None and len(aligned_boxes) > 0:
                # Use the first (most confident) detection
                box = aligned_boxes[0]
                x, y, w, h = box.astype(int).tolist()
                
                keypoints = {}
                if aligned_landmarks is not None and len(aligned_landmarks) > 0:
                    kp = aligned_landmarks[0]
                    keypoints = {
                        'left_eye': kp[0].astype(int).tolist(),
                        'right_eye': kp[1].astype(int).tolist(),
                        'nose': kp[2].astype(int).tolist(),
                        'mouth_left': kp[3].astype(int).tolist(),
                        'mouth_right': kp[4].astype(int).tolist()
                    }

            # Extract nose tip coordinates for centering (paper specification)
            nose_tip = keypoints.get('nose')
            if nose_tip is not None:
                # Use nose tip as center (paper requirement)
                center_x, center_y = nose_tip[0], nose_tip[1]
            else:
                # Fallback to bbox center if nose detection fails
                center_x, center_y = x + w//2, y + h//2
                print("Warning: Nose tip not detected, using bbox center")

            # Calculate crop size: 1.25 * max(height, width) as per paper
            size = int(max(w, h) * self.margin)

            # Crop around nose tip center
            x1 = max(0, center_x - size//2)
            y1 = max(0, center_y - size//2)
            x2 = min(aligned_frame.shape[1], center_x + size//2)
            y2 = min(aligned_frame.shape[0], center_y + size//2)

            # Ensure square crop
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w != crop_h:
                target_size = min(crop_w, crop_h)
                x1 = max(0, center_x - target_size//2)
                y1 = max(0, center_y - target_size//2)
                x2 = min(aligned_frame.shape[1], x1 + target_size)
                y2 = min(aligned_frame.shape[0], y1 + target_size)
                
                if x2 - x1 < target_size:
                    x1 = max(0, x2 - target_size)
                if y2 - y1 < target_size:
                    y1 = max(0, y2 - target_size)

            face = aligned_frame[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                return cv2.resize(face, (300, 300))
            else:
                return None

        except Exception as e:
            print(f"Face detection error: {e}")
            return None

    
    def extract_sequence(self, video_path: str, target_length: int = 6) -> List[np.ndarray]:
        """Extract face sequence from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < target_length:
            cap.release()
            return []
        
        # Sample frames evenly
        # Sample consecutive frames (as per paper requirement)
        if total_frames >= target_length:
            # Start from a random position to get target_length consecutive frames
            start_idx = np.random.randint(0, total_frames - target_length + 1)
            frame_indices = np.arange(start_idx, start_idx + target_length)
        else:
            # If not enough frames, take all available
            frame_indices = np.arange(total_frames)
        faces = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.detect_face(frame_rgb)
                
                if face is not None:
                    faces.append(face)
                else:
                    # If face detection fails, use previous face or create dummy
                    if faces:
                        faces.append(faces[-1])
                    else:
                        faces.append(np.zeros((300, 300, 3), dtype=np.uint8))
        
        cap.release()
        return faces

def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_acc']

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    auc = 0.0
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            pass
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

"""Training pipeline for YOLO pose model with contrastive learning."""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import LOGGER
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict
import json
import os

from ..models.embedding_yolo import ContrastiveLoss, TripletLoss


class PosePairMiner:
    """Mine positive/negative pairs from pose data for contrastive learning."""
    
    def __init__(self, 
                 temporal_window=10,      # frames for temporal consistency
                 position_threshold=0.05,  # normalized distance for spatial consistency
                 angle_threshold=15,       # degrees for pose similarity
                 min_confidence=0.3):
        self.temporal_window = temporal_window
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self.min_confidence = min_confidence
    
    def mine_pairs_from_sequence(self, detections: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Mine positive/negative pairs from a sequence of detections.
        
        Args:
            detections: List of detection dicts with 'frame', 'bbox', 'keypoints', 'confidence'
        
        Returns:
            positive_pairs: [(frame_i, frame_j), ...] - temporally/spatially consistent
            negative_pairs: [(frame_i, frame_j), ...] - inconsistent poses
        """
        positive_pairs = []
        negative_pairs = []
        
        # Filter high-confidence detections
        valid_detections = [
            (i, det) for i, det in enumerate(detections) 
            if det.get('confidence', 0) > self.min_confidence and det.get('keypoints') is not None
        ]
        
        if len(valid_detections) < 2:
            return positive_pairs, negative_pairs
        
        # Mine pairs
        for i, (idx_i, det_i) in enumerate(valid_detections):
            for j, (idx_j, det_j) in enumerate(valid_detections[i+1:], i+1):
                frame_diff = abs(det_i['frame'] - det_j['frame'])
                
                # Temporal proximity check
                if frame_diff <= self.temporal_window:
                    # Check spatial and pose consistency
                    if self._are_poses_consistent(det_i, det_j):
                        positive_pairs.append((idx_i, idx_j))
                    else:
                        # Close in time but different poses = hard negative
                        negative_pairs.append((idx_i, idx_j))
                
                elif frame_diff > self.temporal_window * 2:
                    # Distant in time = likely negative
                    if not self._are_poses_consistent(det_i, det_j):
                        negative_pairs.append((idx_i, idx_j))
        
        return positive_pairs, negative_pairs
    
    def _are_poses_consistent(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections are consistent in pose."""
        kpts1 = np.array(det1['keypoints']).reshape(-1, 2)  # [N, 2]
        kpts2 = np.array(det2['keypoints']).reshape(-1, 2)
        
        if len(kpts1) < 3 or len(kpts2) < 3:
            return False
        
        # Check position consistency (nose position)
        nose1, nose2 = kpts1[0], kpts2[0]  # Assuming first keypoint is nose
        position_dist = np.linalg.norm(nose1 - nose2)
        
        # Normalize by image dimensions (assuming 640x640 for now)
        position_dist_norm = position_dist / 640.0
        
        if position_dist_norm > self.position_threshold:
            return False
        
        # Check head angle consistency
        if len(kpts1) >= 3 and len(kpts2) >= 3:
            angle1 = self._compute_head_angle(kpts1)
            angle2 = self._compute_head_angle(kpts2)
            
            if angle1 is not None and angle2 is not None:
                angle_diff = abs(angle1 - angle2)
                # Handle angle wraparound
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                if angle_diff > self.angle_threshold:
                    return False
        
        return True
    
    def _compute_head_angle(self, keypoints: np.ndarray) -> Optional[float]:
        """Compute head orientation angle from keypoints."""
        if len(keypoints) < 3:
            return None
        
        nose, ear_left, ear_right = keypoints[0], keypoints[1], keypoints[2]
        
        # Compute head vector (from ear midpoint to nose)
        ear_center = (ear_left + ear_right) / 2
        head_vector = nose - ear_center
        
        # Compute angle
        angle = np.arctan2(head_vector[1], head_vector[0]) * 180 / np.pi
        return angle % 360


class ContrastivePoseTrainer(PoseTrainer):
    """Extended YOLO Pose trainer with contrastive learning."""
    
    def __init__(self, cfg='yolo11n-pose.yaml', overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        
        # Contrastive learning components
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.triplet_loss = TripletLoss(margin=0.2)
        self.pair_miner = PosePairMiner()
        
        # Loss weights
        self.contrastive_weight = getattr(self.args, 'contrastive_weight', 0.1)
        self.triplet_weight = getattr(self.args, 'triplet_weight', 0.05)
        
        # Sequence buffer for mining
        self.sequence_buffer = defaultdict(list)
        self.max_sequence_length = 50
        
        LOGGER.info(f"🧠 Contrastive learning enabled - weights: contrastive={self.contrastive_weight}, triplet={self.triplet_weight}")
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model with embedding heads."""
        # For now, use standard YOLO pose model
        # TODO: Implement actual embedding head integration
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        
        # Add embedding dimension as model attribute for future use
        model.embedding_dim = getattr(self.args, 'embedding_dim', 128)
        
        return model
    
    def preprocess_batch(self, batch):
        """Preprocess batch and update sequence buffer."""
        batch = super().preprocess_batch(batch)
        
        # Extract sequence information for pair mining
        if 'im_file' in batch:
            for i, im_file in enumerate(batch['im_file']):
                # Extract sequence ID from filename (e.g., "seq001_frame0123.jpg")
                seq_id = self._extract_sequence_id(im_file)
                frame_num = self._extract_frame_number(im_file)
                
                if seq_id and frame_num is not None:
                    detection_info = {
                        'frame': frame_num,
                        'batch_idx': i,
                        'im_file': im_file
                    }
                    
                    # Add to sequence buffer
                    self.sequence_buffer[seq_id].append(detection_info)
                    
                    # Limit buffer size
                    if len(self.sequence_buffer[seq_id]) > self.max_sequence_length:
                        self.sequence_buffer[seq_id].pop(0)
        
        return batch
    
    def _extract_sequence_id(self, im_file: str) -> Optional[str]:
        """Extract sequence ID from image filename."""
        # Example: "/path/to/seq001_frame0123.jpg" -> "seq001"
        filename = Path(im_file).stem
        if '_frame' in filename:
            return filename.split('_frame')[0]
        return None
    
    def _extract_frame_number(self, im_file: str) -> Optional[int]:
        """Extract frame number from image filename."""
        # Example: "/path/to/seq001_frame0123.jpg" -> 123
        filename = Path(im_file).stem
        if '_frame' in filename:
            try:
                return int(filename.split('_frame')[1])
            except ValueError:
                pass
        return None
    
    def criterion(self, preds, batch):
        """Compute loss with contrastive learning."""
        # For now, use standard pose loss
        # TODO: Add contrastive learning when embedding head is properly implemented
        pose_loss = super().criterion(preds, batch)
        
        # Placeholder for contrastive loss
        if self.training and hasattr(preds, 'embeddings'):
            # Future: add contrastive loss here
            contrastive_loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
            total_loss = pose_loss + self.contrastive_weight * contrastive_loss
        else:
            total_loss = pose_loss
        
        return total_loss
    
    def _compute_contrastive_loss(self, embeddings: List[torch.Tensor], batch: Dict) -> torch.Tensor:
        """Compute contrastive loss from embeddings and batch information."""
        if not embeddings or len(embeddings) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Take embeddings from the largest feature map (usually first one)
        emb = embeddings[0] if isinstance(embeddings, list) else embeddings
        
        if emb.size(0) < 2:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        
        # Mine pairs from current batch sequences
        positive_pairs, negative_pairs = self._mine_batch_pairs(batch)
        
        # Compute contrastive loss
        contrastive = self.contrastive_loss(emb, positive_pairs, negative_pairs)
        
        # Compute triplet loss if enough samples
        triplet = self._compute_triplet_loss(emb, positive_pairs, negative_pairs)
        
        total_contrastive = (
            self.contrastive_weight * contrastive + 
            self.triplet_weight * triplet
        )
        
        return total_contrastive
    
    def _mine_batch_pairs(self, batch: Dict) -> Tuple[List[Tuple], List[Tuple]]:
        """Mine positive/negative pairs from current batch."""
        positive_pairs = []
        negative_pairs = []
        
        if 'im_file' not in batch:
            return positive_pairs, negative_pairs
        
        # Group by sequence
        seq_groups = defaultdict(list)
        for i, im_file in enumerate(batch['im_file']):
            seq_id = self._extract_sequence_id(im_file)
            frame_num = self._extract_frame_number(im_file)
            if seq_id and frame_num is not None:
                seq_groups[seq_id].append((i, frame_num))
        
        # Mine pairs within each sequence
        for seq_id, items in seq_groups.items():
            if len(items) < 2:
                continue
            
            # Sort by frame number
            items.sort(key=lambda x: x[1])
            
            # Create temporal pairs
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    idx_i, frame_i = items[i]
                    idx_j, frame_j = items[j]
                    frame_diff = abs(frame_j - frame_i)
                    
                    if frame_diff <= 5:  # Close frames = positive
                        positive_pairs.append((idx_i, idx_j))
                    elif frame_diff > 20:  # Distant frames = negative
                        negative_pairs.append((idx_i, idx_j))
        
        return positive_pairs, negative_pairs
    
    def _compute_triplet_loss(self, embeddings: torch.Tensor, positive_pairs: List[Tuple], 
                            negative_pairs: List[Tuple]) -> torch.Tensor:
        """Compute triplet loss from mined pairs."""
        if len(positive_pairs) == 0 or len(negative_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Create triplets: (anchor, positive, negative)
        triplets = []
        
        for (a1, p1) in positive_pairs[:10]:  # Limit to avoid memory issues
            for (a2, n1) in negative_pairs[:5]:
                if a1 == a2:  # Same anchor
                    triplets.append((a1, p1, n1))
                    break
        
        if not triplets:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute triplet loss
        total_loss = 0.0
        for anchor_idx, pos_idx, neg_idx in triplets:
            if anchor_idx < len(embeddings) and pos_idx < len(embeddings) and neg_idx < len(embeddings):
                anchor = embeddings[anchor_idx:anchor_idx+1]
                positive = embeddings[pos_idx:pos_idx+1]
                negative = embeddings[neg_idx:neg_idx+1]
                
                loss = self.triplet_loss(anchor, positive, negative)
                total_loss += loss
        
        if len(triplets) > 0:
            return total_loss / len(triplets)
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


def train_embedding_pose_model(
    data_yaml: str,
    model_cfg: str = 'yolo11n-pose.yaml',
    epochs: int = 100,
    batch_size: int = 16,
    embedding_dim: int = 128,
    contrastive_weight: float = 0.1,
    triplet_weight: float = 0.05,
    project: str = 'runs/pose_embedding',
    name: str = 'train',
    **kwargs
):
    """Train YOLO pose model with contrastive learning."""
    
    # Override args for embedding training
    overrides = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'embedding_dim': embedding_dim,
        'contrastive_weight': contrastive_weight,
        'triplet_weight': triplet_weight,
        'project': project,
        'name': name,
        **kwargs
    }
    
    # Create trainer
    trainer = ContrastivePoseTrainer(cfg=model_cfg, overrides=overrides)
    
    # Train
    LOGGER.info(f"🚀 Starting embedding pose training - embedding_dim={embedding_dim}")
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Example training configuration
    train_embedding_pose_model(
        data_yaml='../data/pogona_head_pose.yaml',
        epochs=100,
        batch_size=8,
        embedding_dim=64,
        contrastive_weight=0.1,
        project='../output/models/embedding_pose',
        name='head_pose_v1'
    )
"""Embedding-enhanced pose model for temporal consistency and gap filling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
from dataclasses import dataclass

from .pogona_pose import PogonaHeadPoseModel, ModelOutput
from ..core import HeadPose, PoseKeypoints


@dataclass
class EmbeddingOutput:
    """Container for pose + embedding outputs."""
    pose_output: Optional[ModelOutput]
    embedding: Optional[torch.Tensor]  # Shape: (embedding_dim,)
    confidence: float
    filled_from_embedding: bool = False


class SimpleEmbeddingHead(nn.Module):
    """Lightweight embedding head for pose features."""
    
    def __init__(self, input_dim: int = 1024, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Extract embedding from pose features."""
        return F.normalize(self.head(features), p=2, dim=-1)


class EmbeddingMemory:
    """Efficient circular buffer for recent embeddings and poses."""
    
    def __init__(self, max_size: int = 30, similarity_threshold: float = 0.7):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.embeddings = deque(maxlen=max_size)
        self.poses = deque(maxlen=max_size)
        self.frame_indices = deque(maxlen=max_size)
        self.confidences = deque(maxlen=max_size)
    
    def add(self, embedding: torch.Tensor, pose: ModelOutput, frame_idx: int, conf: float):
        """Add new embedding-pose pair to memory."""
        self.embeddings.append(embedding.cpu())
        self.poses.append(pose)
        self.frame_indices.append(frame_idx)
        self.confidences.append(conf)
    
    def find_similar(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[ModelOutput, float]]:
        """Find most similar poses based on embedding similarity."""
        if not self.embeddings:
            return []
        
        # Compute similarities
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=-1)
        similarities = []
        
        for i, emb in enumerate(self.embeddings):
            emb_norm = F.normalize(emb.unsqueeze(0), p=2, dim=-1)
            sim = torch.cosine_similarity(query_norm, emb_norm).item()
            if sim > self.similarity_threshold:
                similarities.append((i, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, sim in similarities[:top_k]:
            results.append((self.poses[i], sim))
        
        return results
    
    def interpolate_pose(self, similar_poses: List[Tuple[ModelOutput, float]]) -> Optional[ModelOutput]:
        """Interpolate pose from similar embeddings weighted by similarity."""
        if not similar_poses:
            return None
        
        # Weight by similarity
        total_weight = sum(sim for _, sim in similar_poses)
        if total_weight == 0:
            return None
        
        # Weighted average of keypoints and boxes
        weighted_boxes = None
        weighted_keypoints = None
        weighted_confs = None
        
        for pose, sim in similar_poses:
            weight = sim / total_weight
            
            if pose.boxes is not None:
                if weighted_boxes is None:
                    weighted_boxes = pose.boxes * weight
                else:
                    weighted_boxes += pose.boxes * weight
            
            if pose.keypoints is not None:
                if weighted_keypoints is None:
                    weighted_keypoints = pose.keypoints * weight
                else:
                    weighted_keypoints += pose.keypoints * weight
            
            if pose.confs is not None:
                if weighted_confs is None:
                    weighted_confs = pose.confs * weight
                else:
                    weighted_confs += pose.confs * weight
        
        return ModelOutput(
            boxes=weighted_boxes,
            confs=weighted_confs,
            keypoints=weighted_keypoints
        )


class EmbeddingEnhancedPoseModel:
    """Pose model enhanced with embeddings for temporal consistency."""
    
    def __init__(
        self,
        base_model: PogonaHeadPoseModel,
        embedding_dim: int = 64,
        memory_size: int = 30,
        min_confidence: float = 0.3,
        enable_gap_filling: bool = True
    ):
        self.base_model = base_model
        self.embedding_head = SimpleEmbeddingHead(embedding_dim=embedding_dim)
        self.memory = EmbeddingMemory(max_size=memory_size)
        self.min_confidence = min_confidence
        self.enable_gap_filling = enable_gap_filling
        self.frame_count = 0
        
        # Moving average for stability
        self.alpha = 0.7  # EMA factor
        self.last_stable_pose = None
    
    def extract_features(self, numpy_frame: np.ndarray) -> Optional[torch.Tensor]:
        """Extract features from frame using base model's backbone."""
        # This is a simplified feature extractor
        # In practice, you'd hook into YOLO's backbone features
        with torch.no_grad():
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(numpy_frame).float().permute(2, 0, 1).unsqueeze(0)
            frame_tensor = frame_tensor / 255.0
            
            # Dummy feature extraction (replace with actual YOLO backbone hook)
            features = torch.randn(1, 1024)  # This should be real backbone features
            return features.squeeze(0)
    
    def predict_with_embeddings(self, numpy_frame: np.ndarray) -> EmbeddingOutput:
        """Enhanced prediction with embedding-based gap filling."""
        self.frame_count += 1
        
        # Get base pose prediction
        pose_output = self.base_model._extract(numpy_frame)
        
        # Extract features for embedding
        features = self.extract_features(numpy_frame)
        embedding = None
        
        if features is not None:
            embedding = self.embedding_head(features.unsqueeze(0)).squeeze(0)
        
        # Determine if we have a good detection
        has_good_detection = (
            pose_output is not None and 
            pose_output.confs is not None and 
            len(pose_output.confs) > 0 and 
            pose_output.confs.max() > self.min_confidence
        )
        
        if has_good_detection:
            # Good detection - add to memory and return
            if embedding is not None:
                self.memory.add(embedding, pose_output, self.frame_count, pose_output.confs.max())
            
            # Apply EMA smoothing
            if self.last_stable_pose is not None and pose_output.keypoints is not None:
                smoothed_keypoints = (
                    self.alpha * pose_output.keypoints + 
                    (1 - self.alpha) * self.last_stable_pose.keypoints
                )
                pose_output = ModelOutput(
                    boxes=pose_output.boxes,
                    confs=pose_output.confs,
                    keypoints=smoothed_keypoints
                )
            
            self.last_stable_pose = pose_output
            
            return EmbeddingOutput(
                pose_output=pose_output,
                embedding=embedding,
                confidence=pose_output.confs.max() if pose_output.confs is not None else 0.0,
                filled_from_embedding=False
            )
        
        # Poor/no detection - try embedding-based recovery
        elif self.enable_gap_filling and embedding is not None:
            similar_poses = self.memory.find_similar(embedding, top_k=3)
            
            if similar_poses:
                interpolated_pose = self.memory.interpolate_pose(similar_poses)
                
                if interpolated_pose is not None:
                    # Apply stronger EMA for interpolated poses
                    if self.last_stable_pose is not None and interpolated_pose.keypoints is not None:
                        smoothed_keypoints = (
                            0.5 * interpolated_pose.keypoints + 
                            0.5 * self.last_stable_pose.keypoints
                        )
                        interpolated_pose = ModelOutput(
                            boxes=interpolated_pose.boxes,
                            confs=interpolated_pose.confs,
                            keypoints=smoothed_keypoints
                        )
                    
                    avg_similarity = sum(sim for _, sim in similar_poses) / len(similar_poses)
                    
                    return EmbeddingOutput(
                        pose_output=interpolated_pose,
                        embedding=embedding,
                        confidence=avg_similarity * 0.8,  # Reduce confidence for interpolated
                        filled_from_embedding=True
                    )
        
        # Fallback - return original (possibly None) detection
        return EmbeddingOutput(
            pose_output=pose_output,
            embedding=embedding,
            confidence=pose_output.confs.max() if pose_output and pose_output.confs is not None else 0.0,
            filled_from_embedding=False
        )
    
    def reset_memory(self):
        """Reset the embedding memory (e.g., between videos)."""
        self.memory = EmbeddingMemory(max_size=self.memory.max_size)
        self.last_stable_pose = None
        self.frame_count = 0


def create_embedding_enhanced_model(
    weights_path: str,
    embedding_dim: int = 64,
    enable_gap_filling: bool = True,
    **kwargs
) -> EmbeddingEnhancedPoseModel:
    """Factory function to create embedding-enhanced pose model."""
    base_model = PogonaHeadPoseModel(weights_path, **kwargs)
    return EmbeddingEnhancedPoseModel(
        base_model=base_model,
        embedding_dim=embedding_dim,
        enable_gap_filling=enable_gap_filling
    )
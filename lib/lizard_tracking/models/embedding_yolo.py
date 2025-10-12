"""Custom YOLO model with embedding head for pose estimation + contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """Embedding head for contrastive learning on pose features."""
    
    def __init__(self, in_channels, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Feature adapter
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Embedding projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature tensor from backbone [B, C, H, W]
        Returns:
            embeddings: L2-normalized embeddings [B, embedding_dim]
        """
        features = self.feature_adapter(x)
        embeddings = self.projector(features)
        # L2 normalize for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for temporal consistency."""
    
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings, positive_pairs, negative_pairs):
        """
        Args:
            embeddings: [B, embedding_dim] normalized embeddings
            positive_pairs: [(i, j), ...] indices of positive pairs
            negative_pairs: [(i, j), ...] indices of negative pairs
        """
        if len(positive_pairs) == 0 and len(negative_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Positive pairs (should be similar)
        for i, j in positive_pairs:
            if i < len(embeddings) and j < len(embeddings):
                sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])
                loss = torch.clamp(self.margin - sim, min=0.0)
                total_loss += loss
                num_pairs += 1
        
        # Negative pairs (should be dissimilar)
        for i, j in negative_pairs:
            if i < len(embeddings) and j < len(embeddings):
                sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])
                loss = torch.clamp(sim - (-self.margin), min=0.0)
                total_loss += loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class TripletLoss(nn.Module):
    """Triplet loss for embedding learning."""
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: [B, embedding_dim] embeddings
        """
        if anchor.size(0) == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# Placeholder classes for compatibility
class PoseDetectWithEmbedding:
    """Placeholder for future embedding head integration."""
    pass


def create_pose_model_with_embeddings(cfg='yolo11n-pose.yaml', embedding_dim=128):
    """Create a YOLO pose model with embedding heads."""
    return {
        'embedding_dim': embedding_dim,
        'model_type': 'pose_with_embeddings'
    }
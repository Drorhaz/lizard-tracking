#!/usr/bin/env python3
"""Fast embedding training script - adds embedding head to existing YOLO model."""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import json

# Add lib to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
lib_dir = project_root / "lib"
sys.path.insert(0, str(lib_dir))

from lizard_tracking.models.embedding_pose import SimpleEmbeddingHead, EmbeddingMemory

def extract_pose_features(yolo_model, image_path):
    """Extract POSE COORDINATES + VISIBILITY (9 values) for embedding."""
    results = yolo_model(image_path, verbose=False)
    
    if results and len(results[0].keypoints.data) > 0:
        # Get the pose keypoints with visibility: [nose_x, nose_y, nose_v, ear1_x, ear1_y, ear1_v, ear2_x, ear2_y, ear2_v]
        kpts = results[0].keypoints.data[0].cpu().numpy()  # Shape: (3, 3) for x,y,visibility
        
        # Get image dimensions for normalization
        img_height, img_width = results[0].orig_shape
        
        # Normalize coordinates to [0, 1] range for better training
        normalized_kpts = kpts.copy()
        normalized_kpts[:, 0] /= img_width   # Normalize x coordinates
        normalized_kpts[:, 1] /= img_height  # Normalize y coordinates
        # Keep visibility flags as is (already 0-1 range)
        
        # Return flattened features: [nose_x, nose_y, nose_v, ear1_x, ear1_y, ear1_v, ear2_x, ear2_y, ear2_v]
        return torch.tensor(normalized_kpts.flatten(), dtype=torch.float32)  # Shape: (9,)
    
    return torch.zeros(9, dtype=torch.float32)  # No detection found

def compute_contrastive_loss(embeddings, labels, temperature=0.1):
    """Simple contrastive loss for similar poses."""
    # Normalize embeddings
    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
    
    # Create positive pairs (same label)
    batch_size = embeddings.size(0)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()
    
    # Remove diagonal (self-similarity)
    mask = mask - torch.eye(batch_size, device=embeddings.device)
    
    # Compute contrastive loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Mean over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    
    return loss

def train_enhanced_pose():
    """Train embedding head on top of existing YOLO model for gap filling."""
    
    print("🚀 Fast Embedding Training")
    print("=" * 40)
    print("🎯 IMPORTANT: Embeddings are computed on POSE COORDINATES")
    print("   Flow: Image → YOLO → Pose Keypoints → Embedding Vector")
    print("   NOT: Image → Embedding (we embed pose shapes, not raw pixels)")
    print()
    
    # Configuration
    base_model_path = project_root / "output/models/head_pose/best.pt"
    dataset_dir = project_root / "dataset/embedding_dataset"
    output_dir = project_root / "output/models/head_embeddings"
    
    embedding_dim = 64
    learning_rate = 0.001
    epochs = 20  # Fast training!
    batch_size = 8
    
    print(f"📁 Base model: {base_model_path}")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🧠 Embedding dim: {embedding_dim}")
    print(f"⚡ Epochs: {epochs} (fast mode)")
    print()
    
    # Check if base model exists
    if not base_model_path.exists():
        print(f"❌ Base model not found: {base_model_path}")
        print("   Please train a base YOLO model first")
        sys.exit(1)
    
    # Check dataset
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model (frozen for feature extraction)
    print("🔧 Loading base YOLO model...")
    yolo_model = YOLO(str(base_model_path))
    
    # Freeze YOLO parameters (we only train the embedding head)
    for param in yolo_model.model.parameters():
        param.requires_grad = False
    
    print("✅ YOLO model loaded and frozen")
    
    # Create embedding head (input: 9 values from YOLO → output: embedding_dim vector)
    print("🧠 Creating embedding head...")
    embedding_head = SimpleEmbeddingHead(input_dim=9, embedding_dim=embedding_dim)  # 3 keypoints * 3 values (x,y,v)
    optimizer = optim.Adam(embedding_head.parameters(), lr=learning_rate)
    
    # Get training images
    images_dir = dataset_dir / "images"
    image_files = list(images_dir.rglob("*.jpg"))[:100]  # Use first 100 for fast training
    
    print(f"📊 Training on {len(image_files)} images")
    print()
    
    # Training loop
    print("🏋️ Starting embedding training...")
    embedding_head.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Simple batch processing
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_files = image_files[i:i+batch_size]
            
            # Extract features and create embeddings
            batch_features = []
            batch_labels = []
            
            for img_file in batch_files:
                features = extract_pose_features(yolo_model, str(img_file))
                if features.sum() > 0:  # Only use images with detected poses
                    batch_features.append(features)
                    batch_labels.append(0)  # All same class for now
            
            if len(batch_features) < 2:
                continue
            
            # Convert to tensors
            features_tensor = torch.stack(batch_features)
            labels_tensor = torch.tensor(batch_labels)
            
            # Generate embeddings
            embeddings = embedding_head(features_tensor)
            
            # Compute contrastive loss
            loss = compute_contrastive_loss(embeddings, labels_tensor)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Save embedding head
    embedding_path = output_dir / "embedding_head.pt"
    torch.save({
        'embedding_head_state_dict': embedding_head.state_dict(),
        'embedding_dim': embedding_dim,
        'base_model_path': str(base_model_path),
        'input_dim': 9,  # 3 keypoints * 3 values (x, y, visibility)
    }, embedding_path)
    
    print(f"💾 Embedding head saved: {embedding_path}")
    
    # Create combined model info
    model_info = {
        'base_model': str(base_model_path),
        'embedding_head': str(embedding_path),
        'embedding_dim': embedding_dim,
        'input_dim': 9,  # 3 keypoints * 3 values (x, y, visibility)
        'training_epochs': epochs,
        'training_images': len(image_files),
        'purpose': 'gap_filling_and_temporal_consistency'
    }
    
    info_path = output_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"📋 Model info saved: {info_path}")
    print()
    print("🎯 Embedding model ready for gap filling!")
    print("   Usage in arena_mock_app:")
    print(f"   - Base YOLO model: {base_model_path}")
    print(f"   - Embedding head: {embedding_path}")
    print("   - Input: 9D pose features (x,y,visibility for 3 keypoints)")
    print("   - Output: 64D embedding vector")
    print("   - When detection fails → find similar embedding → recover pose")
    print()
    print("✅ Training complete!")

if __name__ == "__main__":
    train_enhanced_pose()
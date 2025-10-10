"""Automatic data mining and preparation for contrastive pose learning."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import os
import shutil
from collections import defaultdict
import math
import sys

# Add lib to path for imports
lib_path = Path(__file__).parent.parent / 'lib'
if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))

class PoseDataMiner:
    """Mine and prepare pose data for contrastive learning."""
    
    def __init__(self, 
                 dataset_dir: str,
                 output_dir: str,
                 sequence_length: int = 30,
                 temporal_gap: int = 5,
                 min_confidence: float = 0.3,
                 augment_hard_negatives: bool = True):
        
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.temporal_gap = temporal_gap
        self.min_confidence = min_confidence
        self.augment_hard_negatives = augment_hard_negatives
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'labels').mkdir(exist_ok=True)
        (self.output_dir / 'sequences').mkdir(exist_ok=True)
        
        print(f"🔍 Initialized PoseDataMiner:")
        print(f"   Dataset: {self.dataset_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Temporal gap: {self.temporal_gap}")
    
    def mine_sequences_from_videos(self, video_paths: List[str], frame_skip: int = 5):
        """Extract sequences from videos with YOLO inference."""
        from ultralytics import YOLO
        
        # Load your trained model
        model_path = self.dataset_dir.parent / 'output/models/head_pose/best.pt'
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        model = YOLO(str(model_path))
        print(f"📊 Loaded model: {model_path}")
        
        sequence_data = []
        global_frame_id = 0
        
        for video_path in video_paths:
            video_path = Path(video_path)
            if not video_path.exists():
                print(f"⚠️ Video not found: {video_path}")
                continue
            
            print(f"🎥 Processing video: {video_path.name}")
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_id = 0
            sequence_id = video_path.stem
            
            with tqdm(total=total_frames//frame_skip, desc=f"Extracting {video_path.name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_id % frame_skip == 0:
                        # Run inference
                        results = model(frame, conf=self.min_confidence, verbose=False)
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            # Extract detection data
                            for i, (box, kpts) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                                if kpts is not None and len(kpts.xy[0]) >= 3:
                                    # Save frame
                                    frame_filename = f"{sequence_id}_frame{global_frame_id:06d}.jpg"
                                    frame_path = self.output_dir / 'images' / frame_filename
                                    cv2.imwrite(str(frame_path), frame)
                                    
                                    # Convert to YOLO format
                                    h, w = frame.shape[:2]
                                    bbox = box.xywhn[0].cpu().numpy()  # normalized xywh
                                    keypoints = kpts.xyn[0].cpu().numpy()  # normalized xy
                                    confidence = box.conf[0].cpu().item()
                                    
                                    # Save label
                                    label_filename = f"{sequence_id}_frame{global_frame_id:06d}.txt"
                                    label_path = self.output_dir / 'labels' / label_filename
                                    self._save_yolo_label(label_path, bbox, keypoints, confidence)
                                    
                                    # Store sequence data
                                    sequence_data.append({
                                        'sequence_id': sequence_id,
                                        'frame_id': global_frame_id,
                                        'video_frame': frame_id,
                                        'timestamp': frame_id / fps,
                                        'filename': frame_filename,
                                        'bbox': bbox.tolist(),
                                        'keypoints': keypoints.flatten().tolist(),
                                        'confidence': confidence,
                                        'video_path': str(video_path)
                                    })
                                    
                                    global_frame_id += 1
                                    break  # Take only first detection per frame
                        
                        pbar.update(1)
                    
                    frame_id += 1
            
            cap.release()
        
        # Save sequence metadata
        sequence_df = pd.DataFrame(sequence_data)
        sequence_df.to_csv(self.output_dir / 'sequence_metadata.csv', index=False)
        print(f"✅ Extracted {len(sequence_data)} frames from {len(video_paths)} videos")
        
        return sequence_data
    
    def _save_yolo_label(self, label_path: Path, bbox: np.ndarray, keypoints: np.ndarray, confidence: float):
        """Save YOLO format label file."""
        with open(label_path, 'w') as f:
            # Class 0 (lizard head)
            line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            
            # Add keypoints (x1 y1 v1 x2 y2 v2 x3 y3 v3)
            for i in range(len(keypoints)):
                if i % 2 == 0:  # x coordinate
                    line += f" {keypoints[i]:.6f}"
                else:  # y coordinate
                    line += f" {keypoints[i]:.6f} 2"  # visibility = 2 (visible)
            
            f.write(line + "\n")
    
    def mine_from_existing_dataset(self):
        """Mine sequences from existing labeled dataset."""
        images_dir = self.dataset_dir / 'images'
        labels_dir = self.dataset_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ Dataset structure not found in {self.dataset_dir}")
            return []
        
        print(f"🔍 Mining sequences from existing dataset...")
        
        # Load all labels
        detections = []
        for label_file in tqdm(list(labels_dir.glob('*.txt')), desc="Loading labels"):
            image_file = images_dir / f"{label_file.stem}.jpg"
            if not image_file.exists():
                continue
            
            detection = self._parse_yolo_label(label_file, image_file)
            if detection:
                detections.append(detection)
        
        print(f"📊 Loaded {len(detections)} detections")
        
        # Group by sequence (if filenames contain sequence info)
        sequences = self._group_detections_by_sequence(detections)
        print(f"📈 Found {len(sequences)} sequences")
        
        # Mine positive/negative pairs
        mined_data = self._mine_contrastive_pairs(sequences)
        
        # Create training splits
        train_data, val_data = self._create_training_splits(mined_data, split_ratio=0.8)
        
        # Save mined data
        self._save_mined_data(train_data, val_data)
        
        return mined_data
    
    def _parse_yolo_label(self, label_file: Path, image_file: Path) -> Optional[Dict]:
        """Parse YOLO label file."""
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
            
            # Parse first detection only
            parts = lines[0].strip().split()
            if len(parts) < 5:
                return None
            
            # Extract bbox
            cls, x, y, w, h = map(float, parts[:5])
            
            # Extract keypoints if available
            keypoints = []
            if len(parts) > 5:
                # Parse keypoints (x1 y1 v1 x2 y2 v2 ...)
                kpt_data = parts[5:]
                for i in range(0, len(kpt_data), 3):
                    if i + 2 < len(kpt_data):
                        kx, ky, kv = map(float, kpt_data[i:i+3])
                        keypoints.extend([kx, ky])
            
            # Extract sequence info from filename
            filename = image_file.stem
            sequence_id, frame_num = self._extract_sequence_info(filename)
            
            return {
                'sequence_id': sequence_id,
                'frame_num': frame_num,
                'filename': image_file.name,
                'bbox': [x, y, w, h],
                'keypoints': keypoints,
                'image_path': str(image_file),
                'label_path': str(label_file)
            }
        
        except Exception as e:
            print(f"⚠️ Error parsing {label_file}: {e}")
            return None
    
    def _extract_sequence_info(self, filename: str) -> Tuple[str, int]:
        """Extract sequence ID and frame number from filename."""
        # Try different patterns
        if '_frame' in filename:
            # Pattern: seq001_frame0123
            parts = filename.split('_frame')
            seq_id = parts[0]
            try:
                frame_num = int(parts[1])
            except ValueError:
                frame_num = 0
        elif '_' in filename:
            # Pattern: video1_123
            parts = filename.split('_')
            seq_id = parts[0]
            try:
                frame_num = int(parts[-1])
            except ValueError:
                frame_num = 0
        else:
            # Single sequence
            seq_id = 'default'
            try:
                frame_num = int(''.join(filter(str.isdigit, filename)))
            except ValueError:
                frame_num = 0
        
        return seq_id, frame_num
    
    def _group_detections_by_sequence(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Group detections by sequence ID."""
        sequences = defaultdict(list)
        
        for detection in detections:
            seq_id = detection['sequence_id']
            sequences[seq_id].append(detection)
        
        # Sort each sequence by frame number
        for seq_id in sequences:
            sequences[seq_id].sort(key=lambda x: x['frame_num'])
        
        return dict(sequences)
    
    def _mine_contrastive_pairs(self, sequences: Dict[str, List[Dict]]) -> Dict:
        """Mine positive/negative pairs for contrastive learning."""
        positive_pairs = []
        negative_pairs = []
        hard_negatives = []
        
        print("⛏️ Mining contrastive pairs...")
        
        for seq_id, detections in tqdm(sequences.items(), desc="Mining sequences"):
            if len(detections) < 2:
                continue
            
            # Mine within sequence (temporal consistency)
            seq_positive, seq_negative = self._mine_sequence_pairs(detections)
            positive_pairs.extend(seq_positive)
            negative_pairs.extend(seq_negative)
        
        # Mine across sequences (hard negatives)
        if self.augment_hard_negatives:
            cross_negatives = self._mine_cross_sequence_negatives(sequences)
            hard_negatives.extend(cross_negatives)
        
        print(f"✅ Mined pairs:")
        print(f"   Positive: {len(positive_pairs)}")
        print(f"   Negative: {len(negative_pairs)}")
        print(f"   Hard negatives: {len(hard_negatives)}")
        
        return {
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'hard_negatives': hard_negatives,
            'sequences': sequences
        }
    
    def _mine_sequence_pairs(self, detections: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """Mine pairs within a single sequence."""
        positive_pairs = []
        negative_pairs = []
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                det1, det2 = detections[i], detections[j]
                frame_diff = abs(det1['frame_num'] - det2['frame_num'])
                
                # Temporal consistency check
                if frame_diff <= self.temporal_gap:
                    # Close in time + similar pose = positive
                    if self._are_poses_similar(det1, det2):
                        positive_pairs.append((det1, det2))
                    else:
                        # Close in time but different pose = hard negative
                        negative_pairs.append((det1, det2))
                
                elif frame_diff > self.temporal_gap * 3:
                    # Far in time = likely negative
                    if not self._are_poses_similar(det1, det2):
                        negative_pairs.append((det1, det2))
        
        return positive_pairs, negative_pairs
    
    def _are_poses_similar(self, det1: Dict, det2: Dict, threshold: float = 0.05) -> bool:
        """Check if two poses are similar."""
        if not det1['keypoints'] or not det2['keypoints']:
            return False
        
        kpts1 = np.array(det1['keypoints']).reshape(-1, 2)
        kpts2 = np.array(det2['keypoints']).reshape(-1, 2)
        
        if len(kpts1) < 3 or len(kpts2) < 3:
            return False
        
        # Check nose position similarity
        nose_dist = np.linalg.norm(kpts1[0] - kpts2[0])
        if nose_dist > threshold:
            return False
        
        # Check head angle similarity
        angle1 = self._compute_head_angle(kpts1)
        angle2 = self._compute_head_angle(kpts2)
        
        if angle1 is not None and angle2 is not None:
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff > 20:  # degrees
                return False
        
        return True
    
    def _compute_head_angle(self, keypoints: np.ndarray) -> Optional[float]:
        """Compute head orientation angle."""
        if len(keypoints) < 3:
            return None
        
        nose, ear_left, ear_right = keypoints[0], keypoints[1], keypoints[2]
        ear_center = (ear_left + ear_right) / 2
        head_vector = nose - ear_center
        
        angle = np.arctan2(head_vector[1], head_vector[0]) * 180 / np.pi
        return angle % 360
    
    def _mine_cross_sequence_negatives(self, sequences: Dict[str, List[Dict]]) -> List[Tuple]:
        """Mine hard negatives across different sequences."""
        hard_negatives = []
        seq_list = list(sequences.items())
        
        for i, (seq1_id, seq1_detections) in enumerate(seq_list):
            for j, (seq2_id, seq2_detections) in enumerate(seq_list[i+1:], i+1):
                # Sample random pairs across sequences
                for _ in range(min(10, len(seq1_detections), len(seq2_detections))):
                    det1 = random.choice(seq1_detections)
                    det2 = random.choice(seq2_detections)
                    
                    # Different sequences are likely negatives
                    if not self._are_poses_similar(det1, det2):
                        hard_negatives.append((det1, det2))
        
        return hard_negatives
    
    def _create_training_splits(self, mined_data: Dict, split_ratio: float = 0.8) -> Tuple[Dict, Dict]:
        """Create training and validation splits."""
        sequences = mined_data['sequences']
        seq_ids = list(sequences.keys())
        random.shuffle(seq_ids)
        
        split_idx = int(len(seq_ids) * split_ratio)
        train_seq_ids = set(seq_ids[:split_idx])
        val_seq_ids = set(seq_ids[split_idx:])
        
        # Split pairs based on sequence membership
        def split_pairs(pairs):
            train_pairs = []
            val_pairs = []
            
            for pair in pairs:
                det1, det2 = pair
                seq1, seq2 = det1['sequence_id'], det2['sequence_id']
                
                if seq1 in train_seq_ids and seq2 in train_seq_ids:
                    train_pairs.append(pair)
                elif seq1 in val_seq_ids and seq2 in val_seq_ids:
                    val_pairs.append(pair)
                # Skip cross-split pairs
            
            return train_pairs, val_pairs
        
        train_pos, val_pos = split_pairs(mined_data['positive_pairs'])
        train_neg, val_neg = split_pairs(mined_data['negative_pairs'])
        train_hard, val_hard = split_pairs(mined_data['hard_negatives'])
        
        train_data = {
            'positive_pairs': train_pos,
            'negative_pairs': train_neg,
            'hard_negatives': train_hard,
            'sequence_ids': train_seq_ids
        }
        
        val_data = {
            'positive_pairs': val_pos,
            'negative_pairs': val_neg,
            'hard_negatives': val_hard,
            'sequence_ids': val_seq_ids
        }
        
        return train_data, val_data
    
    def _save_mined_data(self, train_data: Dict, val_data: Dict):
        """Save mined data to files."""
        # Save pair information
        with open(self.output_dir / 'train_pairs.json', 'w') as f:
            json.dump(self._serialize_pairs(train_data), f, indent=2)
        
        with open(self.output_dir / 'val_pairs.json', 'w') as f:
            json.dump(self._serialize_pairs(val_data), f, indent=2)
        
        # Create dataset YAML for training
        self._create_dataset_yaml(train_data, val_data)
        
        print(f"💾 Saved mined data to {self.output_dir}")
    
    def _serialize_pairs(self, data: Dict) -> Dict:
        """Serialize pair data for JSON storage."""
        def serialize_pair(pair):
            det1, det2 = pair
            return {
                'det1': {
                    'filename': det1['filename'],
                    'sequence_id': det1['sequence_id'],
                    'frame_num': det1['frame_num'],
                    'bbox': det1['bbox'],
                    'keypoints': det1['keypoints']
                },
                'det2': {
                    'filename': det2['filename'],
                    'sequence_id': det2['sequence_id'],
                    'frame_num': det2['frame_num'],
                    'bbox': det2['bbox'],
                    'keypoints': det2['keypoints']
                }
            }
        
        return {
            'positive_pairs': [serialize_pair(p) for p in data['positive_pairs']],
            'negative_pairs': [serialize_pair(p) for p in data['negative_pairs']],
            'hard_negatives': [serialize_pair(p) for p in data['hard_negatives']],
            'sequence_ids': list(data['sequence_ids'])
        }
    
    def _create_dataset_yaml(self, train_data: Dict, val_data: Dict):
        """Create dataset YAML file for YOLO training."""
        yaml_content = f"""# Pose dataset with contrastive learning
path: {self.output_dir.absolute()}
train: images  # Train images
val: images    # Val images (same folder, split by sequence)

# Classes
names:
  0: lizard_head

# Keypoints (nose, ear_left, ear_right)
kpt_shape: [3, 2]  # number of keypoints, number of dimensions (2D)

# Contrastive learning pairs
train_pairs: train_pairs.json
val_pairs: val_pairs.json

# Training configuration
contrastive_weight: 0.1
triplet_weight: 0.05
embedding_dim: 64
"""
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(description="Mine pose data for contrastive learning")
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to existing dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for mined data')
    parser.add_argument('--videos', nargs='*', 
                       help='Video files to extract sequences from')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Maximum sequence length for mining')
    parser.add_argument('--temporal_gap', type=int, default=5,
                       help='Frame gap for positive pairs')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='Minimum confidence for detections')
    
    args = parser.parse_args()
    
    # Create miner
    miner = PoseDataMiner(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        temporal_gap=args.temporal_gap,
        min_confidence=args.min_confidence
    )
    
    # Mine from videos if provided
    if args.videos:
        print("🎥 Mining sequences from videos...")
        miner.mine_sequences_from_videos(args.videos)
    
    # Mine from existing dataset
    print("🔍 Mining from existing dataset...")
    mined_data = miner.mine_from_existing_dataset()
    
    print("✅ Data mining complete!")
    print(f"📁 Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
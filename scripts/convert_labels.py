#!/usr/bin/env python3
"""Convert pose labels from format with visibility to format without visibility and create embedding dataset."""

import os
import shutil
from pathlib import Path

def convert_pose_labels(source_dataset_dir, output_dataset_name="embedding_dataset"):
    """Convert pose labels from 14-column to 11-column format and create new dataset."""
    
    source_path = Path(source_dataset_dir)
    labels_dir = source_path / "labels"
    images_dir = source_path / "images"
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Create embedding dataset directory inside the dataset folder
    dataset_parent = source_path.parent  # This should be 'dataset' folder
    embedding_dataset_path = dataset_parent / output_dataset_name
    embedding_labels_dir = embedding_dataset_path / "labels"
    embedding_images_dir = embedding_dataset_path / "images"
    
    # Create directories
    embedding_dataset_path.mkdir(exist_ok=True)
    embedding_labels_dir.mkdir(exist_ok=True)
    embedding_images_dir.mkdir(exist_ok=True)
    
    print(f"� Creating embedding dataset: {embedding_dataset_path}")
    
    # Copy images directory structure
    print(f"📋 Copying images from {images_dir} to {embedding_images_dir}")
    if embedding_images_dir.exists():
        shutil.rmtree(embedding_images_dir)
    shutil.copytree(images_dir, embedding_images_dir)
    
    total_files = 0
    converted_files = 0
    
    # Process all label files recursively
    for label_file in labels_dir.rglob("*.txt"):
        total_files += 1
        
        # Determine corresponding output path in embedding dataset
        relative_path = label_file.relative_to(labels_dir)
        output_label_file = embedding_labels_dir / relative_path
        
        # Create output directory if needed
        output_label_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            file_converted = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    converted_lines.append(line)
                    continue
                
                parts = line.split()
                
                if len(parts) == 14:  # Has visibility flags
                    # Format: class x y w h x1 y1 v1 x2 y2 v2 x3 y3 v3
                    # Convert to: class x y w h x1 y1 x2 y2 x3 y3
                    new_parts = parts[:5]  # class + bbox
                    
                    # Extract keypoints without visibility
                    for i in range(3):  # 3 keypoints
                        kpt_start = 5 + i * 3
                        if kpt_start + 1 < len(parts):
                            new_parts.append(parts[kpt_start])      # x
                            new_parts.append(parts[kpt_start + 1])  # y
                            # Skip visibility flag (parts[kpt_start + 2])
                    
                    converted_lines.append(' '.join(new_parts))
                    file_converted = True
                
                elif len(parts) == 11:  # Already correct format
                    converted_lines.append(line)
                
                else:
                    print(f"⚠️ Unexpected format in {label_file}: {len(parts)} columns")
                    converted_lines.append(line)
            
            # Write converted labels to embedding dataset
            with open(output_label_file, 'w') as f:
                f.write('\n'.join(converted_lines) + '\n')
            
            if file_converted:
                converted_files += 1
        
        except Exception as e:
            print(f"❌ Error processing {label_file}: {e}")
    
    # Copy or create YAML config file for embedding dataset
    source_yaml = None
    for yaml_file in source_path.glob("*.yaml"):
        source_yaml = yaml_file
        break
    
    if source_yaml:
        embedding_yaml = embedding_dataset_path / source_yaml.name
        shutil.copy2(source_yaml, embedding_yaml)
        print(f"📋 Copied config: {embedding_yaml}")
    
    print(f"✅ Embedding dataset creation complete!")
    print(f"   Source dataset: {source_path}")
    print(f"   Embedding dataset: {embedding_dataset_path}")
    print(f"   Total files: {total_files}")
    print(f"   Converted: {converted_files}")


if __name__ == "__main__":
    # Use dataset/pose as the source
    dataset_dir = "../dataset/pose"
    convert_pose_labels(dataset_dir)
#!/usr/bin/env python3
"""Convert pose labels from format with visibility to format without visibility."""

import os
import shutil
from pathlib import Path

def convert_pose_labels(dataset_dir, backup_suffix="_backup_with_visibility"):
    """Convert pose labels from 14-column to 11-column format."""
    
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / "labels"
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Create backup
    backup_dir = dataset_path / f"labels{backup_suffix}"
    if not backup_dir.exists():
        print(f"📦 Creating backup: {backup_dir}")
        shutil.copytree(labels_dir, backup_dir)
    
    total_files = 0
    converted_files = 0
    
    # Process all label files recursively
    for label_file in labels_dir.rglob("*.txt"):
        total_files += 1
        
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
            
            if file_converted:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(converted_lines) + '\n')
                converted_files += 1
        
        except Exception as e:
            print(f"❌ Error processing {label_file}: {e}")
    
    print(f"✅ Conversion complete!")
    print(f"   Total files: {total_files}")
    print(f"   Converted: {converted_files}")
    print(f"   Backup saved to: {backup_dir}")


if __name__ == "__main__":
    dataset_dir = "/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/dataset"
    convert_pose_labels(dataset_dir)
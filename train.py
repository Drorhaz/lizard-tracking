#!/usr/bin/env python3
"""
Pogona Head Detection Training Script
Train YOLOv11s for lizard head detection
"""

from ultralytics import YOLO
import os

def main():
    # Load a YOLOv11 model
    model = YOLO('yolo11s.pt')  # load pretrained model
    
    # Train the model
    results = model.train(
        data='configs/pogona_head.yaml',
        epochs=10,
        imgsz=640,
        batch=8,   # smaller batch for stability
        device='mps',  # use Metal Performance Shaders on Mac (or 'cpu' if no GPU)
        # Hyperparameters for high recall
        conf=0.25,  # confidence threshold
        iou=0.45,   # IoU threshold for NMS
        # Early stopping
        patience=10,  # epochs to wait for improvement
        # Optimization
        optimizer='AdamW',
        lr0=0.001,  # initial learning rate
        # Data augmentation
        augment=True,
        # Validation
        val=True,
        plots=True,
        save=True,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Validation metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    return results

if __name__ == '__main__':
    main()
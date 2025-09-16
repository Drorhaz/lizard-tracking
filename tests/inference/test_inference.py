#!/usr/bin/env python3
"""
Inference Performance Test for Baseline Lizard Detection Model
Tests speed, accuracy, and robustness on validation images.
"""

from ultralytics import YOLO
import time
import os
import glob
from pathlib import Path
import statistics

def test_model_inference():
    print("🚀 Baseline Model Inference Performance Test")
    print("=" * 60)
    
    # Load baseline model
    model_path = "experiments/sanity_check_cpu/best.pt"
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    print(f"📊 Architecture: YOLOv11s ({model.info()[1]:,} parameters)")
    print()
    
    # Get validation images
    val_images_dir = "dataset/labeled/images/val"
    if not os.path.exists(val_images_dir):
        print(f"❌ Validation images not found at: {val_images_dir}")
        return
    
    image_files = glob.glob(os.path.join(val_images_dir, "*.jpg"))[:20]  # Test on first 20 images
    print(f"🖼️  Found {len(image_files)} validation images for testing")
    print()
    
    # Warmup run
    print("🔥 Warming up model...")
    if image_files:
        warmup_result = model(image_files[0], verbose=False)
        print("✅ Warmup complete")
    print()
    
    # Performance metrics storage
    inference_times = []
    detection_counts = []
    confidence_scores = []
    
    print("⚡ Running inference performance test...")
    print("-" * 40)
    
    for i, image_path in enumerate(image_files, 1):
        # Time the inference
        start_time = time.time()
        results = model(image_path, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract results
        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0
        
        # Get confidence scores
        confidences = []
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy().tolist()
            confidence_scores.extend(confidences)
        
        # Store metrics
        inference_times.append(inference_time)
        detection_counts.append(num_detections)
        
        # Progress update
        image_name = os.path.basename(image_path)
        print(f"Image {i:2d}/20: {image_name[:30]:30} | "
              f"Time: {inference_time:6.1f}ms | "
              f"Detections: {num_detections:2d} | "
              f"Conf: {max(confidences) if confidences else 0:5.3f}")
    
    print("-" * 40)
    print()
    
    # Calculate statistics
    print("📈 INFERENCE PERFORMANCE SUMMARY")
    print("=" * 40)
    
    # Timing statistics
    avg_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    print(f"⏱️  TIMING PERFORMANCE:")
    print(f"   Average inference time: {avg_time:.1f} ms")
    print(f"   Median inference time:  {median_time:.1f} ms")
    print(f"   Fastest inference:      {min_time:.1f} ms")
    print(f"   Slowest inference:      {max_time:.1f} ms")
    print(f"   Theoretical FPS:        {1000/avg_time:.1f} fps")
    print()
    
    # Detection statistics
    total_detections = sum(detection_counts)
    images_with_detections = sum(1 for count in detection_counts if count > 0)
    
    print(f"🎯 DETECTION PERFORMANCE:")
    print(f"   Total detections:       {total_detections}")
    print(f"   Images with detections: {images_with_detections}/{len(image_files)} ({images_with_detections/len(image_files)*100:.1f}%)")
    print(f"   Avg detections per img: {total_detections/len(image_files):.1f}")
    print()
    
    # Confidence statistics
    if confidence_scores:
        avg_conf = statistics.mean(confidence_scores)
        median_conf = statistics.median(confidence_scores)
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        
        print(f"🎯 CONFIDENCE SCORES:")
        print(f"   Average confidence:     {avg_conf:.3f}")
        print(f"   Median confidence:      {median_conf:.3f}")
        print(f"   Lowest confidence:      {min_conf:.3f}")
        print(f"   Highest confidence:     {max_conf:.3f}")
        print()
    
    # Performance assessment
    print("🏆 PERFORMANCE ASSESSMENT")
    print("=" * 40)
    
    # Speed assessment
    if avg_time < 100:
        speed_grade = "🚀 Excellent"
    elif avg_time < 200:
        speed_grade = "⚡ Good"
    elif avg_time < 500:
        speed_grade = "🚶 Acceptable"
    else:
        speed_grade = "🐌 Slow"
    
    print(f"Speed Grade: {speed_grade} ({avg_time:.1f}ms avg)")
    
    # Detection rate assessment
    detection_rate = images_with_detections / len(image_files)
    if detection_rate > 0.9:
        detection_grade = "🎯 Excellent"
    elif detection_rate > 0.8:
        detection_grade = "✅ Good"
    elif detection_rate > 0.6:
        detection_grade = "⚠️  Moderate"
    else:
        detection_grade = "❌ Poor"
    
    print(f"Detection Grade: {detection_grade} ({detection_rate*100:.1f}% images)")
    
    # Confidence assessment
    if confidence_scores:
        if avg_conf > 0.8:
            conf_grade = "💪 High Confidence"
        elif avg_conf > 0.6:
            conf_grade = "👍 Good Confidence"
        elif avg_conf > 0.4:
            conf_grade = "⚠️  Moderate Confidence"
        else:
            conf_grade = "❌ Low Confidence"
        
        print(f"Confidence Grade: {conf_grade} ({avg_conf:.3f} avg)")
    
    print()
    print("🎉 Inference performance test complete!")
    
    # Recommendations
    print()
    print("💡 DEPLOYMENT RECOMMENDATIONS")
    print("=" * 40)
    
    if avg_time < 200:
        print("✅ Model is suitable for real-time applications")
        if avg_time < 50:
            print("🚀 Excellent for high-FPS video processing")
        else:
            print("⚡ Good for standard video processing")
    else:
        print("⚠️  Model may be too slow for real-time video")
        print("💡 Consider model optimization or GPU acceleration")
    
    if detection_rate > 0.8:
        print("✅ High detection rate - ready for production")
    else:
        print("⚠️  Lower detection rate - may need more training data")
    
    return {
        'avg_inference_time': avg_time,
        'detection_rate': detection_rate,
        'avg_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
        'total_detections': total_detections
    }

if __name__ == '__main__':
    test_model_inference()
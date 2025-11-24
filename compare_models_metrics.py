#!/usr/bin/env python3
"""
Compare 3D object detection models across datasets using multiple metrics.

Models compared:
1. PointPillars (KITTI)
2. PointPillars (nuScenes)
3. 3DSSD (KITTI)
4. CenterPoint (nuScenes)
"""

import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

# Model configurations
MODELS = {
    'PointPillars (KITTI)': {
        'pred_file': 'outputs/kitti_pointpillars_gpu/000008_predictions.json',
        'config': 'checkpoints/kitti_pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py',
        'checkpoint': 'checkpoints/kitti_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',
        'dataset': 'kitti',
        'input_path': 'data/kitti/training',
        'frame': '000008',
        'device': 'cuda:0'
    },
    'PointPillars (nuScenes)': {
        'pred_file': 'outputs/nuscenes_pointpillars/sample.pcd_predictions.json',
        'config': 'checkpoints/nuscenes_pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py',
        'checkpoint': 'checkpoints/nuscenes_pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth',
        'dataset': 'any',
        'input_path': 'data/nuscenes_demo/lidar/sample.pcd.bin',
        'frame': None,
        'device': 'cuda:0'
    },
    '3DSSD (KITTI)': {
        'pred_file': 'outputs/3dssd/000008_predictions.json',
        'config': 'checkpoints/3dssd/3dssd_4x4_kitti-3d-car.py',
        'checkpoint': 'checkpoints/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth',
        'dataset': 'kitti',
        'input_path': 'data/kitti/training',
        'frame': '000008',
        'device': 'cuda:0'
    },
    'CenterPoint (nuScenes)': {
        'pred_file': 'outputs/nuscenes_centerpoint/sample.pcd_predictions.json',
        'config': 'checkpoints/nuscenes_centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
        'checkpoint': 'checkpoints/nuscenes_centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth',
        'dataset': 'any',
        'input_path': 'data/nuscenes_demo/lidar/sample.pcd.bin',
        'frame': None,
        'device': 'cuda:0'
    }
}


def load_predictions(pred_file: str) -> Optional[Dict]:
    """Load predictions from JSON file."""
    if not os.path.exists(pred_file):
        return None
    try:
        with open(pred_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None


def calculate_detection_metrics(predictions: Dict) -> Dict:
    """Calculate metrics from predictions."""
    if predictions is None:
        return {}
    
    scores = predictions.get('scores_3d', [])
    labels = predictions.get('labels_3d', [])
    bboxes = predictions.get('bboxes_3d', [])
    
    if not scores:
        return {}
    
    scores_array = np.array(scores)
    
    metrics = {
        'num_detections': len(scores),
        'mean_score': float(np.mean(scores_array)),
        'std_score': float(np.std(scores_array)),
        'min_score': float(np.min(scores_array)),
        'max_score': float(np.max(scores_array)),
        'median_score': float(np.median(scores_array)),
        'high_conf_count': int(np.sum(scores_array >= 0.7)),
        'medium_conf_count': int(np.sum((scores_array >= 0.5) & (scores_array < 0.7))),
        'low_conf_count': int(np.sum(scores_array < 0.5)),
    }
    
    # Calculate score distribution percentiles
    if len(scores_array) > 0:
        metrics['p25_score'] = float(np.percentile(scores_array, 25))
        metrics['p75_score'] = float(np.percentile(scores_array, 75))
        metrics['p90_score'] = float(np.percentile(scores_array, 90))
    
    return metrics


def measure_inference_time(model_config: Dict, num_runs: int = 3) -> Optional[float]:
    """Measure average inference time for a model."""
    if not os.path.exists(model_config['config']) or not os.path.exists(model_config['checkpoint']):
        return None
    
    try:
        import subprocess
        import time
        
        # Build command
        cmd = [
            'python', 'mmdet3d_inference2.py',
            '--dataset', model_config['dataset'],
            '--input-path', model_config['input_path'],
            '--model', model_config['config'],
            '--checkpoint', model_config['checkpoint'],
            '--out-dir', 'outputs/temp_benchmark',
            '--device', model_config['device'],
            '--headless',
            '--score-thr', '0.2'
        ]
        
        if model_config['frame']:
            cmd.extend(['--frame-number', model_config['frame']])
        
        times = []
        for i in range(num_runs):
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - start
            if result.returncode == 0:
                times.append(elapsed)
            else:
                print(f"Warning: Inference failed for {model_config['config']}")
        
        if times:
            return np.mean(times)
    except Exception as e:
        print(f"Error measuring inference time: {e}")
    
    return None


def get_gpu_memory_usage() -> Optional[Dict]:
    """Get GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'total_gb': memory_total,
            'utilization_pct': (memory_allocated / memory_total) * 100
        }
    except:
        return None


def format_table(results: Dict) -> str:
    """Format results as a table."""
    headers = [
        'Model',
        'Dataset',
        'Detections',
        'Mean Score',
        'Max Score',
        'High Conf (>=0.7)',
        'Inference Time (s)',
        'FPS',
        'GPU Mem (GB)',
        'Score Std'
    ]
    
    # Calculate column widths
    col_widths = [max(len(h), 20) for h in headers]
    for model_name, metrics in results.items():
        col_widths[0] = max(col_widths[0], len(model_name))
    
    # Create separator
    sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    
    # Build table
    lines = [sep]
    
    # Header
    header_row = '| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)) + ' |'
    lines.append(header_row)
    lines.append(sep)
    
    # Data rows
    for model_name, metrics in results.items():
        dataset = metrics.get('dataset', 'N/A')
        num_det = metrics.get('num_detections', 0)
        mean_score = metrics.get('mean_score', 0.0)
        max_score = metrics.get('max_score', 0.0)
        std_score = metrics.get('std_score', 0.0)
        high_conf = metrics.get('high_conf_count', 0)
        inf_time = metrics.get('inference_time', None)
        fps = metrics.get('fps', None)
        gpu_mem = metrics.get('gpu_memory', None)
        
        row = [
            model_name,
            dataset,
            str(num_det),
            f"{mean_score:.3f}",
            f"{max_score:.3f}",
            str(high_conf),
            f"{inf_time:.3f}" if inf_time else "N/A",
            f"{fps:.1f}" if fps else "N/A",
            f"{gpu_mem:.2f}" if gpu_mem else "N/A",
            f"{std_score:.3f}"
        ]
        
        data_row = '| ' + ' | '.join(r.ljust(w) for r, w in zip(row, col_widths)) + ' |'
        lines.append(data_row)
        lines.append(sep)
    
    return '\n'.join(lines)


def main():
    """Main comparison function."""
    print("=" * 80)
    print("3D Object Detection Model Comparison")
    print("=" * 80)
    print("\nLoading predictions and calculating metrics...\n")
    
    results = {}
    
    for model_name, config in MODELS.items():
        print(f"Processing {model_name}...")
        
        # Load predictions
        predictions = load_predictions(config['pred_file'])
        
        if predictions is None:
            print(f"  Warning: No predictions found for {model_name}")
            continue
        
        # Calculate detection metrics
        metrics = calculate_detection_metrics(predictions)
        metrics['dataset'] = config['dataset'].upper()
        metrics['model_name'] = model_name
        
        # Measure inference time (disabled by default - uncomment to enable)
        # print(f"  Measuring inference time (this may take a moment)...")
        # inf_time = measure_inference_time(config, num_runs=2)
        # if inf_time:
        #     metrics['inference_time'] = inf_time
        #     metrics['fps'] = 1.0 / inf_time if inf_time > 0 else 0
        # else:
        #     metrics['inference_time'] = None
        #     metrics['fps'] = None
        metrics['inference_time'] = None
        metrics['fps'] = None
        
        # Get GPU memory (measure before and after inference)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before measurement
            gpu_mem_before = get_gpu_memory_usage()
            if gpu_mem_before:
                metrics['gpu_memory'] = gpu_mem_before['allocated_gb']
                metrics['gpu_memory_pct'] = gpu_mem_before['utilization_pct']
        else:
            metrics['gpu_memory'] = None
            metrics['gpu_memory_pct'] = None
        
        results[model_name] = metrics
    
    # Print detailed metrics
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    print()
    
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Dataset: {metrics.get('dataset', 'N/A')}")
        print(f"  Number of Detections: {metrics.get('num_detections', 0)}")
        print(f"  Score Statistics:")
        print(f"    Mean: {metrics.get('mean_score', 0):.3f}")
        print(f"    Std:  {metrics.get('std_score', 0):.3f}")
        print(f"    Min:  {metrics.get('min_score', 0):.3f}")
        print(f"    Max:  {metrics.get('max_score', 0):.3f}")
        print(f"    Median: {metrics.get('median_score', 0):.3f}")
        print(f"  Confidence Distribution:")
        print(f"    High (>=0.7):   {metrics.get('high_conf_count', 0)}")
        print(f"    Medium (0.5-0.7): {metrics.get('medium_conf_count', 0)}")
        print(f"    Low (<0.5):    {metrics.get('low_conf_count', 0)}")
        if 'p25_score' in metrics:
            print(f"  Percentiles:")
            print(f"    P25: {metrics['p25_score']:.3f}")
            print(f"    P75: {metrics['p75_score']:.3f}")
            print(f"    P90: {metrics['p90_score']:.3f}")
        print()
    
    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(format_table(results))
    print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if results:
        all_detections = [m.get('num_detections', 0) for m in results.values()]
        all_mean_scores = [m.get('mean_score', 0) for m in results.values()]
        all_high_conf = [m.get('high_conf_count', 0) for m in results.values()]
        
        print(f"Total Detections Across All Models: {sum(all_detections)}")
        print(f"Average Detections per Model: {np.mean(all_detections):.1f}")
        print(f"Average Mean Score: {np.mean(all_mean_scores):.3f}")
        print(f"Total High Confidence Detections (>=0.7): {sum(all_high_conf)}")
        print()
        
        # Best performers
        best_mean_score = max(results.items(), key=lambda x: x[1].get('mean_score', 0))
        most_detections = max(results.items(), key=lambda x: x[1].get('num_detections', 0))
        most_high_conf = max(results.items(), key=lambda x: x[1].get('high_conf_count', 0))
        
        print("Best Performers:")
        print(f"  Highest Mean Score: {best_mean_score[0]} ({best_mean_score[1].get('mean_score', 0):.3f})")
        print(f"  Most Detections: {most_detections[0]} ({most_detections[1].get('num_detections', 0)})")
        print(f"  Most High Confidence: {most_high_conf[0]} ({most_high_conf[1].get('high_conf_count', 0)})")
    
    print()
    print("=" * 80)
    print("NOTES")
    print("=" * 80)
    print("• Metrics Calculated: Detection counts, score statistics, confidence distribution")
    print("• Inference Time & FPS: Can be enabled by uncommenting measurement code")
    print("• GPU Memory: Shows current allocated memory (may vary)")
    print("• mAP/AP: Requires ground truth annotations for calculation")
    print("• Precision/Recall: Requires ground truth annotations for calculation")
    print("• IoU: Requires ground truth bounding boxes for calculation")
    print("=" * 80)


if __name__ == "__main__":
    main()


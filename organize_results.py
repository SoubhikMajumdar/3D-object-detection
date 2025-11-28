"""Organize all videos and screenshots into a results folder with descriptive names."""
import os
import shutil
from pathlib import Path

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("Creating results folder...")

# Define files to copy with descriptive names
files_to_copy = [
    # Videos
    ("outputs/detections_demo.mp4", "results/demo_video.mp4"),
    ("outputs/detections_demo.gif", "results/demo_video.gif"),
    
    # KITTI PointPillars
    ("outputs/kitti_pointpillars/000008_2d_vis.png", "results/kitti_pointpillars_2d.png"),
    ("outputs/kitti_pointpillars/000008_open3d.png", "results/kitti_pointpillars_3d.png"),
    
    # KITTI PointPillars GPU
    ("outputs/kitti_pointpillars_gpu/000008_2d_vis.png", "results/kitti_pointpillars_gpu_2d.png"),
    
    # KITTI PointPillars CUDA
    ("outputs/kitti_pointpillars_cuda/000008_2d_vis.png", "results/kitti_pointpillars_cuda_2d.png"),
    
    # KITTI PointPillars 3-class
    ("outputs/kitti_pointpillars_3class/000008_2d_vis.png", "results/kitti_pointpillars_3class_2d.png"),
    
    # 3DSSD
    ("outputs/3dssd/000008_2d_vis.png", "results/3dssd_2d.png"),
    ("outputs/3dssd/000008_open3d.png", "results/3dssd_3d.png"),
    
    # 3DSSD Filtered
    ("outputs/3dssd_filtered/000008_2d_vis.png", "results/3dssd_filtered_2d.png"),
    
    # 3DSSD High Threshold
    ("outputs/3dssd_high_thresh/000008_2d_vis.png", "results/3dssd_high_threshold_2d.png"),
    
    # 3DSSD Very High Threshold
    ("outputs/3dssd_very_high/000008_2d_vis.png", "results/3dssd_very_high_threshold_2d.png"),
    
    # nuScenes PointPillars
    ("outputs/nuscenes_pointpillars/sample_open3d.png", "results/nuscenes_pointpillars_3d.png"),
]

copied = 0
skipped = 0

for src, dst in files_to_copy:
    src_path = Path(src)
    dst_path = Path(dst)
    
    if src_path.exists():
        # Create parent directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"  Copied: {src} -> {dst}")
        copied += 1
    else:
        print(f"  Skipped (not found): {src}")
        skipped += 1

print(f"\nDone! Copied {copied} files, skipped {skipped} files.")
print(f"Results folder: {results_dir.absolute()}")


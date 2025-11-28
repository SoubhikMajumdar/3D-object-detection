# 3D Object Detection Results

This folder contains all visualization outputs and demo videos from the 3D object detection inference runs.

## Demo Videos

- **demo_video.mp4** - Complete demo video showing all model visualizations (6 frames, 24 fps)
- **demo_video.gif** - Animated GIF version of the demo video

## KITTI Dataset Results (Frame 000008)

### PointPillars Models

- **kitti_pointpillars_2d.png** - PointPillars 2D visualization (projected bounding boxes on camera image)
- **kitti_pointpillars_3d.png** - PointPillars 3D visualization (Open3D point cloud with bounding boxes)
- **kitti_pointpillars_gpu_2d.png** - PointPillars GPU version 2D visualization
- **kitti_pointpillars_cuda_2d.png** - PointPillars CUDA version 2D visualization
- **kitti_pointpillars_3class_2d.png** - PointPillars 3-class model (Car, Pedestrian, Cyclist) 2D visualization

### 3DSSD Models

- **3dssd_2d.png** - 3DSSD 2D visualization (default score threshold 0.2)
- **3dssd_3d.png** - 3DSSD 3D visualization (Open3D point cloud with bounding boxes)
- **3dssd_filtered_2d.png** - 3DSSD with filtered detections (score threshold 0.6)
- **3dssd_high_threshold_2d.png** - 3DSSD with high score threshold
- **3dssd_very_high_threshold_2d.png** - 3DSSD with very high score threshold

## nuScenes Dataset Results

- **nuscenes_pointpillars_3d.png** - PointPillars 3D visualization on nuScenes sample data

## Notes

- All KITTI results are from frame `000008`
- 2D visualizations show projected 3D bounding boxes on camera images
- 3D visualizations show point clouds with 3D bounding boxes in Open3D format
- Score thresholds affect the number of detections shown (higher thresholds = fewer but more confident detections)


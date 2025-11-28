## Project Overview

This workspace demonstrates 3D object detection inference on sample KITTI and nuScenes frames using MMDetection3D models. It automates artifact export (point clouds, bounding boxes, raw JSON predictions, Open3D screenshots) and compiles a short demo video for comparison.

The core driver is `mmdet3d_inference2.py`, a customized version of OpenMMLab's inference script with enhanced visualization and export utilities. Helper scripts provide KITTI calibration generation and Open3D viewing.

> 📊 **See [REPORT.md](REPORT.md) for comprehensive evaluation results, metrics, and analysis of all models.**

## Prerequisites

1. **Python 3.10** – installed via Microsoft Store (`winget install Python.Python.3.10`).
2. **Virtual environment** – created in the repo root: `py -3.10 -m venv .venv`.
3. **NVIDIA GPU (optional but recommended)** – for CUDA acceleration (GTX 1650 or better recommended).
4. **CUDA Toolkit 11.3+** – for GPU support (PyTorch will use CUDA 11.8 which is compatible).

### Activate Environment (PowerShell)
```powershell
& .\.venv\Scripts\Activate.ps1
```

### Install Dependencies

#### Option 1: CPU-Only Setup
```powershell
python -m pip install -U pip
pip install openmim open3d opencv-python-headless==4.8.1.78 opencv-python==4.8.1.78 \
    matplotlib tqdm moviepy pandas seaborn
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
mim install mmengine
pip install mmcv==2.1.0 mmdet==3.2.0
mim install mmdet3d
```

#### Option 2: CUDA Setup (Recommended for GPU)
```powershell
python -m pip install -U pip
pip install openmim open3d opencv-python-headless==4.8.1.78 opencv-python==4.8.1.78 \
    matplotlib tqdm moviepy pandas seaborn
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
mim install mmengine
pip install mmcv==2.1.0 mmdet==3.2.0
mim install mmdet3d
```

**Verify CUDA Installation:**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

> **Note:** We pin NumPy 1.26.x and OpenCV 4.8.1 to match the prebuilt MMDetection3D sparse ops. Installing in this order prevents ABI conflicts.

## Repository Layout

```
scripts/
  export_kitti_calib.py        # Converts KITTI demo PKL to calib txt
  open3d_view_saved_ply.py     # Local Open3D visualization helper
mmdet3d_inference2.py          # Enhanced MMDetection3D inference script
external/mmdetection3d         # Upstream repo (for sample data/config)
data/                          # Prepared KITTI / nuScenes demo inputs
outputs/                       # All inference artifacts
```

## Initial Data Prep

Demo inputs come from the cloned `external/mmdetection3d/demo/data/` directory. Before running inference:

1. **Copy KITTI sample**
   ```powershell
   Copy-Item external\mmdetection3d\demo\data\kitti\000008.bin data\kitti\training\velodyne\
   Copy-Item external\mmdetection3d\demo\data\kitti\000008.png data\kitti\training\image_2\
   Copy-Item external\mmdetection3d\demo\data\kitti\000008.txt data\kitti\training\label_2\
   python scripts/export_kitti_calib.py `
     external/mmdetection3d/demo/data/kitti/000008.pkl `
     data/kitti/training/calib/000008.txt
   ```

2. **Copy nuScenes sample**
   ```powershell
   Copy-Item external\mmdetection3d\demo\data\nuscenes\*CAM*jpg data\nuscenes_demo\images\
   Copy-Item external\mmdetection3d\demo\data\nuscenes\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin `
     data\nuscenes_demo\lidar\sample.pcd.bin
   ```

## Download Pretrained Models

Use OpenMIM to grab the relevant checkpoints and configs.

```powershell
# PointPillars models
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest checkpoints/kitti_pointpillars
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest checkpoints/kitti_pointpillars_3class
mim download mmdet3d --config pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d --dest checkpoints/nuscenes_pointpillars

# 3DSSD (requires CUDA)
mim download mmdet3d --config 3dssd_4x4_kitti-3d-car --dest checkpoints/3dssd

# CenterPoint (requires CUDA)
mim download mmdet3d --config centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest checkpoints/nuscenes_centerpoint
```

Resulting folders include both the config `.py` and the `.pth` weights used in inference.

## Running Inference

### Device Selection

- **CPU**: Use `--device cpu` for PointPillars models (slower, ~10-12 seconds per frame)
- **CUDA**: Use `--device cuda:0` for all models (faster, recommended if GPU available)

### Available Models

| Model | Dataset | CPU | CUDA | Notes |
|-------|---------|-----|------|-------|
| PointPillars | KITTI | ✅ | ✅ | Works on both, faster on CUDA |
| PointPillars 3-class | KITTI | ✅ | ✅ | Detects Pedestrian, Cyclist, Car |
| PointPillars | nuScenes | ✅ | ✅ | Works on both, faster on CUDA |
| 3DSSD | KITTI | ❌ | ✅ | **Requires CUDA** (furthest point sampling) |
| CenterPoint | nuScenes | ❌ | ✅ | **Requires CUDA** (sparse convolution) |

### 1. KITTI PointPillars (CPU or CUDA)

```powershell
# CPU version
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000008 `
  --model checkpoints\kitti_pointpillars\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py `
  --checkpoint checkpoints\kitti_pointpillars\hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth `
  --out-dir outputs\kitti_pointpillars `
  --device cpu `
  --headless `
  --score-thr 0.2

# CUDA version (faster)
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000008 `
  --model checkpoints\kitti_pointpillars\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py `
  --checkpoint checkpoints\kitti_pointpillars\hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth `
  --out-dir outputs\kitti_pointpillars_gpu `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

### 2. KITTI PointPillars 3-class (CPU or CUDA)

```powershell
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000008 `
  --model checkpoints\kitti_pointpillars_3class\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py `
  --checkpoint checkpoints\kitti_pointpillars_3class\hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth `
  --out-dir outputs\kitti_pointpillars_3class `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

### 3. nuScenes PointPillars (CPU or CUDA)

```powershell
python mmdet3d_inference2.py `
  --dataset any `
  --input-path data\nuscenes_demo\lidar\sample.pcd.bin `
  --model checkpoints\nuscenes_pointpillars\pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py `
  --checkpoint checkpoints\nuscenes_pointpillars\hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth `
  --out-dir outputs\nuscenes_pointpillars `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

### 4. KITTI 3DSSD (CUDA Required)

```powershell
python mmdet3d_inference2.py `
  --dataset kitti `
  --input-path data\kitti\training `
  --frame-number 000008 `
  --model checkpoints\3dssd\3dssd_4x4_kitti-3d-car.py `
  --checkpoint checkpoints\3dssd\3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth `
  --out-dir outputs\3dssd `
  --device cuda:0 `
  --headless `
  --score-thr 0.6
```

> **Note:** 3DSSD produces many false positives. Use `--score-thr 0.6` or `0.7` to reduce them.

### 5. nuScenes CenterPoint (CUDA Required)

```powershell
python mmdet3d_inference2.py `
  --dataset any `
  --input-path data\nuscenes_demo\lidar\sample.pcd.bin `
  --model checkpoints\nuscenes_centerpoint\centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py `
  --checkpoint checkpoints\nuscenes_centerpoint\centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth `
  --out-dir outputs\nuscenes_centerpoint `
  --device cuda:0 `
  --headless `
  --score-thr 0.2
```

### Output Files

All inference runs generate:
- `*_predictions.json` - Raw prediction data (scores, labels, bounding boxes)
- `*_2d_vis.png` - 2D visualization with projected bounding boxes
- `*_points.ply` - Point cloud data (Open3D format)
- `*_pred_bboxes.ply` - Predicted 3D bounding boxes (Open3D format)
- `*_pred_labels.ply` - Predicted labels (Open3D format)
- `*_axes.ply` - Coordinate axes (Open3D format)
- `preds/*.json` - Formatted prediction JSON files

## Open3D Visualization

The helper script supports both interactive and headless viewing.

### Capture Screenshot (headless)
```powershell
python scripts/open3d_view_saved_ply.py --dir outputs\kitti_pointpillars --basename 000008 `
  --width 1600 --height 1200 --save-path outputs\kitti_pointpillars\000008_open3d.png --no-show
```

### Interactive Exploration
```powershell
python scripts/open3d_view_saved_ply.py --dir outputs\kitti_pointpillars --basename 000008 --width 1600 --height 1200
```
- Mouse rotate, right-click pan, scroll zoom, `Q` to close.
- Repeat with `--dir outputs\nuscenes_pointpillars --basename sample.pcd` for nuScenes.

## Demo Video Assembly

A short stitched video (`outputs/detections_demo.mp4`) is produced with MoviePy:

```powershell
python -c "from moviepy import ImageClip, concatenate_videoclips; import os; frames=['outputs/kitti_pointpillars/000008_2d_vis.png','outputs/kitti_pointpillars/000008_open3d.png','outputs/3dssd/000008_2d_vis.png','outputs/3dssd/000008_open3d.png','outputs/kitti_pointpillars_3class/000008_2d_vis.png','outputs/nuscenes_pointpillars/sample_open3d.png']; clips=[ImageClip(f).with_duration(3) for f in frames if os.path.exists(f)]; concatenate_videoclips(clips, method='compose').write_videofile('outputs/detections_demo.mp4', fps=24, codec='libx264', audio=False)"
```

Alternatively, use the helper script:
```powershell
python create_demo.py
```

Inline preview (GIF):

![3D Detection Demo](outputs/detections_demo.gif)

## Runtime & Score Stats

- `outputs/inference_times.json` – measured wall-clock runtime per frame using PowerShell’s `Measure-Command`.
- `outputs/inference_stats.json` – mean/max/min detection scores and raw class counts.
- `outputs/combined_stats.json` – merged view adding runtime and top-three class tallies.

To regenerate stats:

```powershell
python -c "import json, numpy as np; mappings={'kitti':{0:'Car'},'nuscenes':{0:'car',1:'truck',2:'construction_vehicle',3:'bus',4:'trailer',5:'barrier',6:'motorcycle',7:'bicycle',8:'pedestrian',9:'traffic_cone'}}; files={'kitti':'outputs/kitti_pointpillars/000008_predictions.json','nuscenes':'outputs/nuscenes_pointpillars/sample.pcd_predictions.json'}; aggregated={};
for name,path in files.items():
    data=json.load(open(path))
    scores=np.array(data.get('scores_3d', []), dtype=float)
    labels=data.get('labels_3d', [])
    class_map=mappings[name]
    counts={}
    for lab in labels:
        cls=class_map.get(lab, str(lab))
        counts[cls]=counts.get(cls,0)+1
    aggregated[name]={
        'detections': len(labels),
        'mean_score': float(scores.mean()) if scores.size else None,
        'score_std': float(scores.std()) if scores.size else None,
        'max_score': float(scores.max()) if scores.size else None,
        'min_score': float(scores.min()) if scores.size else None,
        'class_counts': counts
    }
json.dump(aggregated, open('outputs/inference_stats.json','w'), indent=2)"
```

## Model Comparison

Compare all models using the comparison script:

```powershell
python compare_models_metrics.py
```

This generates:
- Detailed metrics for each model (detection counts, score statistics)
- Comparison table
- Summary statistics
- Best performer analysis

See `REPORT.md` for comprehensive analysis and results.

## Troubleshooting

### CUDA Issues
- **CUDA not available:** Ensure PyTorch CUDA version matches your CUDA toolkit. Install with `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA out of memory:** Reduce batch size or use CPU for PointPillars models
- **Sparse conv errors:** CenterPoint requires CUDA. Use PointPillars on CPU if GPU unavailable

### Model-Specific Issues
- **3DSSD false positives:** Use higher score threshold (`--score-thr 0.6` or `0.7`)
- **PointPillars low scores on nuScenes:** This is expected; consider filtering with higher threshold
- **CenterPoint/3DSSD CPU errors:** These models require CUDA. Use PointPillars for CPU inference

### General Issues
- **NUMPY ABI errors:** Ensure NumPy 1.26.x remains installed; newer 2.x builds break mmcv's compiled ops
- **Open3D import failures:** Confirm `pip show open3d` inside the active venv
- **Long runtimes:** CPU inference is slow (~10-12s per frame); use CUDA for faster inference
- **Missing checkpoints:** Run `mim download` commands to fetch model weights

## Key Outputs (for reference)

### 2D Visualizations
- `outputs/kitti_pointpillars_gpu/000008_2d_vis.png` - PointPillars (KITTI)
- `outputs/kitti_pointpillars_3class/000008_2d_vis.png` - PointPillars 3-class (KITTI)
- `outputs/3dssd/000008_2d_vis.png` - 3DSSD (KITTI)
- `outputs/nuscenes_centerpoint/` - CenterPoint (nuScenes)

### 3D Visualizations
- `outputs/*/000008_points.ply` - Point clouds
- `outputs/*/000008_pred_bboxes.ply` - 3D bounding boxes
- `outputs/*/000008_pred_labels.ply` - Labels

### Data Files
- `outputs/*/000008_predictions.json` - Raw predictions
- `outputs/detections_demo.mp4` - Demo video (if generated)
- `metrics_output.txt` - Model comparison metrics

## Documentation

- **REPORT.md** - Comprehensive evaluation report with:
  - Setup instructions
  - Model specifications
  - Detailed metrics and results
  - Performance analysis
  - Visualizations and screenshots
  - Conclusions and recommendations


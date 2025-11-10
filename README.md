## Project Overview

This workspace demonstrates 3D object detection inference on sample KITTI and nuScenes frames using MMDetection3D models. It automates artifact export (point clouds, bounding boxes, raw JSON predictions, Open3D screenshots) and compiles a short demo video for comparison.

The core driver is `mmdet3d_inference2.py`, a customized version of OpenMMLab's inference script with enhanced visualization and export utilities. Helper scripts provide KITTI calibration generation and Open3D viewing.

## Prerequisites

1. **Python 3.10** – installed via Microsoft Store (`winget install Python.Python.3.10`).
2. **Virtual environment** – created in the repo root: `py -3.10 -m venv .venv`.
3. **Activate env (PowerShell)**
   ```powershell
   & .\.venv\Scripts\Activate.ps1
   ```
4. **Install dependencies**
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
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest checkpoints/kitti_pointpillars
mim download mmdet3d --config pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d --dest checkpoints/nuscenes_pointpillars
```

Resulting folders include both the config `.py` and the `.pth` weights used in inference.

## Running Inference

### 1. KITTI PointPillars

```powershell
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
```

Outputs saved into `outputs/kitti_pointpillars/` include:
- `*_points.ply`, `*_axes.ply`, `*_pred_bboxes.ply`, `*_pred_labels.ply`
- `*_predictions.json` and `preds/*.json`
- `*_2d_vis.png` (projected boxes) and optional Open3D capture (see below)

### 2. nuScenes PointPillars

```powershell
python mmdet3d_inference2.py `
  --dataset any `
  --input-path data\nuscenes_demo\lidar\sample.pcd.bin `
  --model checkpoints\nuscenes_pointpillars\pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py `
  --checkpoint checkpoints\nuscenes_pointpillars\hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth `
  --out-dir outputs\nuscenes_pointpillars `
  --device cpu `
  --headless `
  --score-thr 0.2
```

> The CenterPoint config cannot run on CPU because sparse conv kernels lack CPU implementations; PointPillars works without CUDA, though inference takes ~10–12 seconds per frame.

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
python -c "from moviepy import ImageClip, concatenate_videoclips; import os; frames=['outputs/kitti_pointpillars/000008_2d_vis.png','outputs/kitti_pointpillars/000008_open3d.png','outputs/nuscenes_pointpillars/sample_open3d.png']; clips=[ImageClip(f).with_duration(3) for f in frames if os.path.exists(f)]; concatenate_videoclips(clips, method='compose').write_videofile('outputs/detections_demo.mp4', fps=24, codec='libx264', audio=False)"
```

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

## Troubleshooting

- **Missing CUDA kernels:** stick with PointPillars or install GPU-enabled PyTorch + mmcv-full.
- **NUMPY ABI errors:** ensure NumPy 1.26.x remains installed; newer 2.x builds break mmcv’s compiled ops.
- **Open3D import failures:** confirm `pip show open3d` inside the active venv.
- **Long runtimes:** CPU inference is slow; for speed, switch to CUDA builds and run on GPU.

## Key Outputs (for reference)

- `outputs/kitti_pointpillars/000008_2d_vis.png`
- `outputs/kitti_pointpillars/000008_open3d.png`
- `outputs/nuscenes_pointpillars/sample_open3d.png`
- `outputs/detections_demo.mp4`
- `outputs/*.ply`, `outputs/*predictions.json`

## Next Steps

- Batch-process multiple frames by setting `--frame-number -1` for KITTI or looping over nuScenes files.
- Integrate evaluation metrics (AP, mAP) by comparing predictions with ground-truth labels.
- Swap in other MMDetection3D configs (SECOND, CenterPoint, etc.) on a GPU-enabled setup.

import argparse
import json
from pathlib import Path


def numpy_to_str(arr):
    return " ".join(f"{x:.12e}" for x in arr)


def save_kitti_calib(pkl_path: Path, output_path: Path):
    import pickle
    import numpy as np

    with open(pkl_path, "rb") as f:
        sample = pickle.load(f)

    data = sample["data_list"][0]
    images = data["images"]

    camera_keys = ["CAM0", "CAM1", "CAM2", "CAM3"]
    # KITTI expects P0..P3 matrices. If a camera is missing, fall back to CAM2.
    proj_mats = {}
    fallback = images.get("CAM2")
    for idx, cam_key in enumerate(camera_keys):
        cam = images.get(cam_key, fallback)
        mat = cam["cam2img"]
        mat = np.asarray(mat, dtype=float)[:3, :4]
        proj_mats[f"P{idx}"] = mat

    r0_rect = np.asarray(images.get("R0_rect"), dtype=float)[:3, :3]

    lidar2cam = np.asarray(images["CAM2"]["lidar2cam"], dtype=float)[:3, :4]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for key in ["P0", "P1", "P2", "P3"]:
            f.write(f"{key}: {numpy_to_str(proj_mats[key].ravel())}\n")
        f.write(f"R0_rect: {numpy_to_str(r0_rect.ravel())}\n")
        f.write(f"Tr_velo_to_cam: {numpy_to_str(lidar2cam.ravel())}\n")
        # Optional identity for imu to velo (not used but common in KITTI files)
        f.write("Tr_imu_to_velo: " + numpy_to_str(np.eye(3, 4).ravel()) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export KITTI calibration from mmdet3d demo pickle")
    parser.add_argument("pkl", type=Path, help="Input PKL file from mmdet3d demo data")
    parser.add_argument("output", type=Path, help="Output KITTI calibration txt path")
    args = parser.parse_args()

    save_kitti_calib(args.pkl, args.output)
    print(f"Saved KITTI calibration to {args.output}")


if __name__ == "__main__":
    main()

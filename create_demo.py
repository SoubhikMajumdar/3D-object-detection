"""Create demo video and GIF from visualization outputs."""
from moviepy import ImageClip, concatenate_videoclips
from PIL import Image
import os

# Define frames to include in the demo
frames = [
    'outputs/kitti_pointpillars/000008_2d_vis.png',
    'outputs/kitti_pointpillars/000008_open3d.png',
    'outputs/3dssd/000008_2d_vis.png',
    'outputs/3dssd/000008_open3d.png',
    'outputs/kitti_pointpillars_3class/000008_2d_vis.png',
    'outputs/nuscenes_pointpillars/sample_open3d.png',
]

# Filter to only existing and valid image files
valid_frames = []
print("Checking frames...")
for f in frames:
    if os.path.exists(f):
        try:
            # Verify image can be opened
            img = Image.open(f)
            img.load()  # Load the image to verify it's valid
            img.close()
            valid_frames.append(f)
            print(f"  OK: {f}")
        except Exception as e:
            print(f"  ERROR: {f} (error: {e})")
    else:
        print(f"  MISSING: {f}")

print(f"\nFound {len(valid_frames)} valid frames to include:")

if not valid_frames:
    print("No valid frames found! Exiting.")
    exit(1)

# Create video clips (3 seconds each)
clips = []
for f in valid_frames:
    try:
        clip = ImageClip(f).with_duration(3)
        clips.append(clip)
    except Exception as e:
        print(f"Warning: Could not load {f}: {e}")

if not clips:
    print("No valid clips created! Exiting.")
    exit(1)

# Concatenate and write video
print(f"\nCreating video from {len(clips)} clips...")
video = concatenate_videoclips(clips, method='compose')
video.write_videofile(
    'outputs/detections_demo.mp4',
    fps=24,
    codec='libx264',
    audio=False,
    logger=None
)
print("Video created: outputs/detections_demo.mp4")

# Create GIF from video
print("\nCreating GIF...")
video.write_gif(
    'outputs/detections_demo.gif',
    fps=24,
    logger=None
)
print("GIF created: outputs/detections_demo.gif")

print("\nDone!")


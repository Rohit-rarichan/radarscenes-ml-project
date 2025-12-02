from pathlib import Path
from loaders.load_camera import load_camera_image, get_camera_image_path

DATASET = Path("/home/rraricha/Datasets/RadarScenes/RadarScenes/data")

seq = "sequence_1"
timestamp = 156862647501     # FIRST entry in scenes.json

img_path = get_camera_image_path(DATASET, seq, timestamp)
print("Image path:", img_path)
print("Exists:", img_path.exists())

img = load_camera_image(img_path)
img.show()
print("Loaded:", img is not None)
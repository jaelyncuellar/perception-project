from pathlib import Path 
import cv2 

folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")

images = sorted(folder.glob("*.png"))
print(f"Found {len(images)} images")

if images: 
    frame = cv2.imread(str(images[0]))
    print(frame.shape if frame is not None else "Failed to read image")
else: 
    print("No images found. Check the dataset folder path.")
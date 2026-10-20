# ORB detection code 
from pathlib import Path 
import cv2 

# 1 - loc dataset 
folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")
images = sorted(folder.glob("*.png"))
if not images: 
    raise FileNotFoundError("No RGB images found!")

# 2 - read 1st img 
frame = cv2.imread(str(images[0]))
if frame is None: 
    raise RuntimeError("Could not load image!")

# 3 - convert to grayscale 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#4 - create orb ft detector 
orb = cv2.ORB_create(nfeatures=1000)

# 5 - detect keypoints & compute descriptors 
keypoints, descriptors = orb.detectAndCompute(gray, None)
print(f"Detected {len(keypoints)} keypoints")
if descriptors is not None: 
    print(f"Descriptor shape: {descriptors.shape}")

#6. draw keypts on og img 
result = cv2.drawKeypoints(
    frame, keypoints, None, color=(0,255,0), 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

#7. save annotated img 
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "orb_keypoints.png"
cv2.imwrite(str(output_file), result)

print(f"Saved result to {output_file}")

#8 - display annotated img 
cv2.imshow("ORB Keypoints", result)
cv2.waitKey(0)
cv2.destroyAllWindows()



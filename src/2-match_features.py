# matches fts (FAST BRIEF) from 2 consec images 
from pathlib import Path 
import cv2

# 1 - loc dataset imgs 
folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")
images = sorted(folder.glob("*.png"))
if len(images) < 2: 
    raise FileNotFoundError("Need at least 2 RGB images to compare & match features.")

#2 - select 2 nearby frames (slight sep helps us observe cam movement)
img1 = cv2.imread(str(images[0]))
img2 = cv2.imread(str(images[30]))

if img1 is None or img2 is None: 
    raise RuntimeError("Could not load images!")

# 3 - convert to grayscale 
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 4 - detect ORB keypts & descriptors 
orb = cv2.ORB_create(nfeatures=1000)
kp1,des1 = orb.detectAndCompute(gray1,None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print(f"Frame 1: {len(kp1)} keypoints")
print(f"Frame 2: {len(kp2)} keypoints")

if des1 is None or des2 is None: 
    raise RuntimeError("No descriptors found in one of the frames!")

# 5 - match descriptors using Hamming distance 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# find 2 nearest matches for each descriptor 
pairs = bf.knnMatch(des1, des2, k=2)

# 6 - filter matches using lowe's ratio test 
good_matches = [] 
ratio_test_threshold = 0.95
for pair in pairs: 
    if len(pair) ==2: 
        best, second_best = pair 
        if best.distance < ratio_test_threshold * second_best.distance: 
            good_matches.append(best)
print(f"Good matches: {len(good_matches)}")

# 7 - visualize the best matches 
good_matches = sorted(good_matches, 
    key=lambda match:match.distance
)

result = cv2.drawMatches(
    img1, kp1, img2, kp2, 
    good_matches[:50], 
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 8 - save visualization 
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "orb_matches.png"

if not cv2.imwrite(str(output_path), result): 
    raise RuntimeError("could not save matching image!")

print(f"Saved matching visualization to {output_path}")

# 9 - display result 
cv2.imshow("ORB feature matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
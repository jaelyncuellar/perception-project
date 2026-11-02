import cv2
import numpy as np
from pathlib import Path

# reuse ORB ft detection & matching code 

# 1. Load two consecutive images
folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")
images = sorted(folder.glob("*.png"))
if not images: 
    raise FileNotFoundError("Error loading dataset images")
img1 = cv2.imread(str(images[0]))
img2 = cv2.imread(str(images[2]))
if img1 is None or img2 is None: 
    raise RuntimeError("Could not load iamges!")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. Detect ORB features
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None: 
    raise RuntimeError("No descriptors found in one of the frames")

# 3. Match descriptors using BFMatcher
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = bf.knnMatch(des1, des2,k=2)

# 4. Apply Lowe's ratio test
good_matches = []
ratio_test_threshold = 0.95
for pair in pairs: 
    if len(pair)==2: 
        best, second_best = pair 
        if best.distance < ratio_test_threshold*second_best.distance: 
            good_matches.append(best)
good_matches=sorted(good_matches, key=lambda match:match.distance)

# 5. Extract pts1 and pts2
result = cv2.drawMatches(img1, kp1, img2, kp2, 
                         good_matches[:50], 
                         None, 
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# extract corresponding ft coords
pts1=np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2=np.float32([kp2[m.trainIdx].pt for m in good_matches])

print("Matched points:", len(pts1))
print("pts1 shape:", pts1.shape)
print("pts2 shape:", pts2.shape)

# 6. Define camera intrinsics K
K = np.array([
    [517.3, 0.0, 318.6],
    [0.0, 516.5, 255.3],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 7. Estimate E using RANSAC
if len(pts1) < 5:
    raise ValueError("Not enough matches to estimate E")

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None or mask is None:
    raise RuntimeError("Essential Matrix estimation failed")


# 8. Print E and inlier statistics. Inspect results
num_inliers = np.count_nonzero(mask)
inlier_ratio = num_inliers / len(pts1)

print("\nEssential Matrix:")
print(E)

print("\nRANSAC results:")
print("Total matches:", len(pts1))
print("Inliers:", num_inliers)
print("Outliers:", len(pts1) - num_inliers)
print(f"Inlier ratio: {inlier_ratio:.2%}")



# 1. Call cv2.recoverPose() using the Essential Matrix, matched coordinates, and camera intrinsics.
# 2. Pass your existing RANSAC inlier mask so pose recovery starts with geometrically consistent matches.
# 3. Capture the recovered rotation matrix, translation vector, pose inlier count, and updated mask.
pose_inliers, R, t, updated_pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask) 

# 4. Print the rotation and translation.
# 5. Print their NumPy shapes.
# 6. Compare the number of pose inliers against your 695 Essential Matrix inliers.
print(f"Rotation: {R}")
print(f"translation: {t}")
print(f"Rotation shape: {R.shape}")
print(f"translation shape: {t.shape}")
print(f"pose inliers: {pose_inliers} vs Essential Matrix inliers: {num_inliers}")


# day 6 - milestone 5: validate estimated pose 
print("\n--- Pose Validation ---")
# challenge 1 - validate rotation matrix 
# compute R transpose multiplied by R 
rotation_check = R.T @ R
# compare against the 3x3 identity matrix 
is_orthogonal = np.allclose(rotation_check, np.eye(3))
# calc determinant & check if ~1 
det_R = np.linalg.det(R)
is_proper_rotation = np.isclose(det_R, 1.0)
print("\nR^T R:")
print(rotation_check)
print("Orthogonal:", is_orthogonal)
print("Determinant:", det_R)
print("Proper rotation:", is_proper_rotation)

# challenge 2 - validate trnaslation magnitude 
# compute euclidean norm of trans vector 
translation_norm = np.linalg.norm(t)
#check if trans normalized 
is_unit_translation = np.isclose(translation_norm, 1.0)
print("\nTranslation Validation:")
print("Translation Magnitude:", translation_norm)
print("Unit translation:", is_unit_translation)

# challenge 3 - compare inlier counts 
# num_inliers was calculated before recoverPose() 
additional_rejected = num_inliers - pose_inliers
pose_inlier_ratio = pose_inliers/len(pts1)

print("\n Inlier Comparison:")
print("Essential Matrix inliers:", num_inliers)
print("Pose inliers:", pose_inliers)
print("Additional matches rejected:", additional_rejected)
print(f"Final pose inlier ratio: {pose_inlier_ratio:.2%}")









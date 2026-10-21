# 1. Finds all .png files inside your dataset's rgb directory.
# 2. Sorts the filenames into the correct order.
# 3. Selects only the first 10 images.
# 4. Loops through those images.
# 5. Loads each image using OpenCV.
# 6. Prints the filename and image dimensions.
# You may reuse code from test_dataset.py. That's normal software engineering practice.


from pathlib import Path # Object Oriented filesystem paths 
    # easier to use path - no more long string
    # also has glob built in - recurisve 
import cv2 # openCV -> open source computer vision lib -> CV, image proc, ML. treats imgs/vids as multidim NumPy arrays 
import numpy as np


folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")
images = sorted(folder.glob("*.png")) 
    # glob finds file & directory pathnames that match a specific pattern. 
    # uses the more simple Unix shell wildcard - not regex 
if not images: 
    raise FileNotFoundError("Error loading dataset images")

# set up objs
orb = cv2.ORB_create(nfeatures=1000)
bf=cv2.BFMatcher(cv2.NORM_HAMMING) # for hamming dist
failed_pairs = []
first_ten_images = images[:10]
MINIMUM_MATCHES = 8000

for i in range(1, len(first_ten_images)): 
    img1 = cv2.imread(str(first_ten_images[i-1])) # using str avoid compat issues with diff cv versions
    img2 = cv2.imread(str(first_ten_images[i]))
    if img1 is None or img2 is None: 
        failed_pairs.append((first_ten_images[i-1], first_ten_images[i]))
        print("Warning: one or two images could not load")
        continue
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Warning: one or two ORB descriptors could not be loaded.")
        failed_pairs.append((first_ten_images[i-1], first_ten_images[i]))
        continue

    pairs = bf.knnMatch(des1, des2, k=2) # find 2 nearest matches for descriptors 

    # lowe's ratio test to filter out weak 
    good_matches = [] 
    frame1_matches = []
    frame2_matches = []

    ratio_test_threshold = 0.75
    for pair in pairs: 
        if len(pair)==2: 
            best, second_best = pair
            if best.distance < ratio_test_threshold * second_best.distance: 
                good_matches.append(best)
                frame1_matches.append(kp1[best.queryIdx].pt)
                frame2_matches.append(kp2[best.trainIdx].pt)

    if len(good_matches) < MINIMUM_MATCHES: 
        print(f"Warning: Less than {MINIMUM_MATCHES} good matches found")
        failed_pairs.append((first_ten_images[i-1], first_ten_images[i]))
        print(f"Frame {i} -> Frame {i+1}")
        print(f"Good matches: {len(good_matches)}")
        print(f"Status: SKIPPED (insufficent matches)")
        continue

    frame1_matches = np.array(frame1_matches, dtype=np.float32)
    frame2_matches = np.array(frame2_matches, dtype=np.float32)

    #verification
    assert frame1_matches.shape == frame2_matches.shape 
    assert frame1_matches.shape[1] == 2

    print(f"Frame {i} -> Frame {i+1}")
    print(f"Keypoints: {len(kp1)} -> {len(kp2)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"Percentage: {len(good_matches)/len(kp1) : .1%}")
    print(f"Points shape: {frame1_matches.shape}")
    print(f"Status: OK")

            
        
        

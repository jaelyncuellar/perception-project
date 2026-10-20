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

folder = Path("data/rgbd_dataset_freiburg1_xyz/rgb")
images = sorted(folder.glob("*.png")) 
    # glob finds file & directory pathnames that match a specific pattern. 
    # uses the more simple Unix shell wildcard - not regex 
if not images: 
    raise FileNotFoundError("Error loading dataset images")

first_ten_images = images[:10]
for i in range(1, len(first_ten_images)): 
    img1 = cv2.imread(first_ten_images[i-1])
    img2 = cv2.imread(first_ten_images[i])
    if img1 is None or img2 is None: 
        raise RuntimeError("Could not load current_iamge")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    print(f"Matching frame {first_ten_images[i-1].name} with {first_ten_images[i].name}")


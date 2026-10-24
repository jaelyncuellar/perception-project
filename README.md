# Perception Project


This project uses Python and OpenCV to analyze phone-camera video and progressively build a monocular visual-odometry prototype that estimates camera movement from visual features. 



## Camera Calibration 
- The instrinsic matrix K is a matrix with camera parameters. It represents the transformation from normalized camera coordinates to 2D pixel coordinates. It combines the camera's focal length (f_x, f_y) and principal point (c_x, c_y) into a single 3x3 matrix. 

- The calibration parameters used in this project come from the published RGB camera calibration for the TUM RGB-D Freiburg 1 dataset. 

- Accurate calibration is important for visual odometry because focal lengths & principal determine how 3D points are projected onto the image plane. Incorrect parameters can introduce errors when estimating camera rotation & translation from matched features.

- Lens distortion matters because real cameras can cause image points to appear at positions different from those predicted by the ideal pinhole camera model. Distortion changes pixel positions and affects feature correspondences. Accounting for distortion helps improve the accuracy of geometric calculations and camera motion estimation. 
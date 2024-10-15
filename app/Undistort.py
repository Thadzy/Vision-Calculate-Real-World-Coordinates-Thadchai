import cv2
import numpy as np
import os

# Load the previously saved calibration data
calibration_file = './calibration_data.npz'

# Ensure the calibration file exists
if os.path.exists(calibration_file):
    data = np.load(calibration_file)
    mtx = data['mtx']
    dist = data['dist']
else:
    print("Calibration data not found. Please run camera calibration first.")
    exit()

# Proceed with the rest of your undistortion code
image_path = './images/saved_img_2.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to read the image '{image_path}'")
    exit()

h, w = img.shape[:2]

# Perform undistortion using the loaded camera matrix and distortion coefficients
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# Optional: Crop the image to the valid region of interest (roi)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y + h, x:x + w]

# Display or save the undistorted image
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./images/undistorted_test_image.jpg', undistorted_img)

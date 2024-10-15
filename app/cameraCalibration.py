#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

# Defining the dimensions of the checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vectors to store 3D points (object points) and 2D points (image points)
objpoints = []
imgpoints = []

# Defining world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Extracting paths of images
images = glob.glob('./images/*.jpg')

# Processing each image
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Image {fname} could not be loaded.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If corners are found, refine the pixel coordinates
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)  # Display the image for 500ms
    else:
        print(f"Corners not found in image {fname}")

cv2.destroyAllWindows()

# Ensure that at least one valid image was processed
if len(objpoints) > 0 and len(imgpoints) > 0:
    # Get the image size from the last processed image
    h, w = gray.shape[:2]

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # Print calibration results
    print("Camera matrix: \n", mtx)
    print("Distortion coefficients: \n", dist)
    print("Rotation vectors: \n", rvecs)
    print("Translation vectors: \n", tvecs)
else:
    print("No valid images were processed for calibration.")

# Save the camera calibration result for future use
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


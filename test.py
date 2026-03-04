import cv2
import glob
import os
 
LEFT_DIR = "calib_dataset/left"
RIGHT_DIR = "calib_dataset/right"
 
lf = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))[0]
rf = sorted(glob.glob(os.path.join(RIGHT_DIR, "right_*.png")))[0]
 
imgL = cv2.imread(lf)
imgR = cv2.imread(rf)
 
print("Left image:", lf, "shape:", imgL.shape)   # (H, W, 3)
print("Right image:", rf, "shape:", imgR.shape)
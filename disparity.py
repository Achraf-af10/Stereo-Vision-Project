import cv2
import glob
import os
import numpy as np
 
LEFT_DIR = "calib_dataset/rectified/left"
RIGHT_DIR = "calib_dataset/rectified/right"
 
def list_pairs():
 
    lefts = sorted(glob.glob(os.path.join(LEFT_DIR, "rect_left_*.png")))
    pairs = []
 
    for lf in lefts:
        idx = os.path.basename(lf).replace("rect_left_", "").replace(".png", "")
        rf = os.path.join(RIGHT_DIR, f"rect_right_{idx}.png")
 
        if os.path.exists(rf):
            pairs.append((lf, rf, idx))
 
    return pairs
 
 
def compute_disparity(imgL, imgR):
 
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
 
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*8,
        blockSize=5,
        P1=8*3*5**2,
        P2=32*3*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
 
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
 
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
 
    return disp_vis
 
 
def main():
 
    pairs = list_pairs()
 
    if len(pairs) == 0:
        print("Aucune paire rectifiée trouvée.")
        return
 
    print("Touches :")
    print("space → image suivante")
    print("q → quitter")
 
    for i,(lf,rf,idx) in enumerate(pairs):
 
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
 
        disparity = compute_disparity(imgL,imgR)
 
        cv2.imshow("Left rectified", imgL)
        cv2.imshow("Right rectified", imgR)
        cv2.imshow("Disparity", disparity)
 
        key = cv2.waitKey(0) & 0xFF
 
        if key == ord('q'):
            break
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
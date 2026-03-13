import cv2
import numpy as np
import glob
import os

LEFT_DIR = "test_disparity/rectified/left"
RIGHT_DIR = "test_disparity/rectified/right"


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

    window_size = 5

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=16 * 16,
        blockSize=window_size,

        P1=9 * 3 * window_size,
        P2=128 * 3 * window_size,

        disp12MaxDiff=12,
        uniquenessRatio=40,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,

        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # WLS FILTER
    lmbda = 70000
    sigma = 1.7

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)

    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filtered = wls_filter.filter(displ, imgL, None, dispr)

    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    filtered = np.uint8(filtered)

    return filtered


def main():

    pairs = list_pairs()

    if len(pairs) == 0:
        print("Aucune paire rectifiée trouvée")
        return

    print("space → image suivante")
    print("q → quitter")

    for lf, rf, idx in pairs:

        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)

        disparity = compute_disparity(imgL, imgR)

        cv2.imshow("Left", imgL)
        cv2.imshow("Right", imgR)
        cv2.imshow("Disparity WLS", disparity)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
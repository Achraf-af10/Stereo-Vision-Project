import os
import glob
import cv2
import numpy as np
from stereo_capture import LEFT_DIR, RIGHT_DIR, CHESSBOARD_SIZE, SQUARE_SIZE

CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
MIN_PAIRS = 10
OUT_DIR = "calib_dataset"

# =========================
# OUTILS
# =========================
def build_object_points(chessboard_size=CHESSBOARD_SIZE, square_size=SQUARE_SIZE):
    nx, ny = chessboard_size
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    objp *= square_size
    return objp

def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4))
    return ok, corners

def calibrate_camera(image_files, name="cam"):
    objp = build_object_points()
    objpoints, imgpoints, used_files = [], [], []
    img_size = None
    for f in image_files:
        img = cv2.imread(f)
        if img is None: continue
        if img_size is None: img_size = (img.shape[1], img.shape[0])
        ok, corners = detect_corners(img)
        if ok:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            used_files.append(f)
    if len(used_files) < 5:
        raise RuntimeError(f"[{name}] Pas assez d'images valides")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None, criteria=CALIB_CRITERIA)

    # Erreurs reprojection
    all_dists = []
    for i in range(len(objpoints)):
        proj,_ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        d = np.linalg.norm(imgpoints[i].reshape(-1,2) - proj.reshape(-1,2), axis=1)
        all_dists.append(d)
    all_dists = np.concatenate(all_dists)
    mean_err = float(np.mean(all_dists))
    rms_err = float(np.sqrt(np.mean(all_dists**2)))
    print(f"\n--- Calibration {name} ---")
    print(f"Images utilisées: {len(used_files)}/{len(image_files)}")
    print("K =\n", K)
    print("dist =", dist.ravel())
    print(f"Mean error = {mean_err:.4f}px | RMS = {rms_err:.4f}px")
    return K, dist, rvecs, tvecs, used_files, mean_err, rms_err

def validate_reprojection(image_files, K, dist, rvecs, tvecs, title):
    objp = build_object_points()
    print(f"\n--- Validation {title} ---")
    print("Vert=coins détectés, Rouge=points reprojetés")
    print("Espace=image suivante | q=quitter")
    i = 0
    while i < len(image_files):
        img = cv2.imread(image_files[i])
        if img is None:
            i += 1
            continue
        ok, corners = detect_corners(img)
        if not ok:
            i += 1
            continue
        proj,_ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        vis = img.copy()
        for p in corners.reshape(-1,2): cv2.circle(vis, tuple(map(int,p)),4,(0,255,0),-1)
        for p in proj.reshape(-1,2): cv2.circle(vis, tuple(map(int,p)),3,(0,0,255),-1)
        cv2.putText(vis, f"{title} | view {i}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.imshow(f"Validation - {title}", vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break
        elif key == 32: i += 1
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
def main():
    left_files = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(RIGHT_DIR, "right_*.png")))

    if len(left_files) < MIN_PAIRS:
        raise RuntimeError("Pas assez de paires capturées")

    K_L, dist_L, rvecs_L, tvecs_L, used_L, mean_L, rms_L = calibrate_camera(left_files, "LEFT")
    K_R, dist_R, rvecs_R, tvecs_R, used_R, mean_R, rms_R = calibrate_camera(right_files, "RIGHT")

    np.savez(os.path.join(OUT_DIR,"intrinsics_left.npz"), K=K_L, dist=dist_L, rvecs=rvecs_L, tvecs=tvecs_L, mean_err=mean_L, rms_err=rms_L)
    np.savez(os.path.join(OUT_DIR,"intrinsics_right.npz"), K=K_R, dist=dist_R, rvecs=rvecs_R, tvecs=tvecs_R, mean_err=mean_R, rms_err=rms_R)
    print(f"Intrinsics saved in {OUT_DIR}/intrinsics_*.npz")

    validate_reprojection(used_L, K_L, dist_L, rvecs_L, tvecs_L, "LEFT")
    validate_reprojection(used_R, K_R, dist_R, rvecs_R, tvecs_R, "RIGHT")

if __name__ == "__main__":
    main()
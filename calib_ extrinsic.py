import os
import glob
import cv2
import numpy as np
import random

# =========================
# CONFIG
# =========================
CHESSBOARD_SIZE = (11, 7)   # coins internes (cols, rows)
SQUARE_SIZE = 0.0265           # m (20 mm)
MIN_INTRINSIC_IMAGES = 5
MIN_STEREO_PAIRS = 8
OUT_DIR = "calib_dataset"
LEFT_DIR = os.path.join(OUT_DIR, "left")
RIGHT_DIR = os.path.join(OUT_DIR, "right")

# Affichage épipolaire
N_RANDOM_POINTS = 5
RANDOM_SEED = 42  # None pour vrai hasard

# OpenCV
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
CALIB_CRITERIA  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# =========================
# UTILITAIRES
# =========================
def build_object_points():
    nx, ny = CHESSBOARD_SIZE
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    objp *= SQUARE_SIZE
    return objp

def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, FIND_FLAGS)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), SUBPIX_CRITERIA)
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
    if len(used_files) < MIN_INTRINSIC_IMAGES:
        raise RuntimeError(f"[{name}] Pas assez d'images valides")
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None, criteria=CALIB_CRITERIA)

    # Erreurs de reprojection
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

    # Sauvegarde
    np.savez(os.path.join(OUT_DIR,f"intrinsics_{name.lower()}.npz"),
             K=K, dist=dist, rvecs=rvecs, tvecs=tvecs,
             mean_err=mean_err, rms_err=rms_err)

    return K, dist, rvecs, tvecs, used_files

def list_stereo_pairs():
    lefts = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))
    pairs = []
    for lf in lefts:
        idx = os.path.basename(lf).replace("left_", "").replace(".png","")
        rf = os.path.join(RIGHT_DIR, f"right_{idx}.png")
        if os.path.exists(rf):
            pairs.append((lf, rf))
    return pairs

def hsv_color(i, n):
    h = int((i / max(1, n-1))*179)
    hsv = np.uint8([[[h,255,255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return tuple(map(int,bgr))

def point_line_distance(x, y, a, b, c):
    return abs(a*x + b*y + c) / (np.sqrt(a*a + b*b)+1e-12)

def draw_line(img, line_abc, color, thickness=2):
    a,b,c = line_abc
    h,w = img.shape[:2]
    eps = 1e-12
    if abs(b)>eps:
        y0 = int(round((-c - a*0)/b))
        y1 = int(round((-c - a*(w-1))/b))
        pt1,pt2 = (0,y0),(w-1,y1)
    else:
        x = int(round(-c/(a+eps)))
        pt1,pt2 = (x,0),(x,h-1)
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

# =========================
# CALIBRATION STÉRÉO EXTRINSÈQUE
# =========================
def stereo_extrinsic_calibration(K_L, dist_L, K_R, dist_R):
    pairs = list_stereo_pairs()
    if len(pairs)<MIN_STEREO_PAIRS:
        raise RuntimeError(f"Pas assez de paires stéréo: {len(pairs)} (min {MIN_STEREO_PAIRS})")
    
    objp = build_object_points()
    objpoints,imgpointsL,imgpointsR,used_pairs = [],[],[],[]
    img_size = None

    for lf, rf in pairs:
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
        if img_size is None:
            img_size = (imgL.shape[1], imgL.shape[0])
        okL, cornersL = detect_corners(imgL)
        okR, cornersR = detect_corners(imgR)
        if okL and okR:
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            used_pairs.append((lf,rf))

    # Calibration stéréo
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        K_L, dist_L, K_R, dist_R,
        img_size,
        criteria=CALIB_CRITERIA,
        flags=flags
    )

    print("\n--- STEREO EXTRINSICS ---")
    print(f"RMS: {rms}")
    print("R =\n", R)
    print("T =\n", T)
    print("E =\n", E)
    print("F =\n", F)

    # Sauvegarde
    np.savez(os.path.join(OUT_DIR,"stereo_extrinsics.npz"),
             R=R,T=T,E=E,F=F,rms=rms,
             K_L=K_L,dist_L=dist_L,K_R=K_R,dist_R=dist_R,
             chessboard_size=np.array(CHESSBOARD_SIZE),
             square_size=float(SQUARE_SIZE),
             used_pairs=np.array(used_pairs,dtype=object))

    print(f"Sauvegardé: {OUT_DIR}/stereo_extrinsics.npz")

    # Visualisation épipolaire
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    n_points = CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1]
    for lf, rf in used_pairs:
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
        okL,cornersL = detect_corners(imgL)
        okR,cornersR = detect_corners(imgR)
        if not(okL and okR): continue
        ptsL = cornersL.reshape(-1,1,2).astype(np.float32)
        ptsR = cornersR.reshape(-1,1,2).astype(np.float32)
        idxs = random.sample(range(n_points),k=min(N_RANDOM_POINTS,n_points))
        selL = ptsL[idxs]; selR = ptsR[idxs]
        linesR = cv2.computeCorrespondEpilines(selL,1,F).reshape(-1,3)
        linesL = cv2.computeCorrespondEpilines(selR,2,F).reshape(-1,3)
        visL, visR = imgL.copy(), imgR.copy()
        for j, idx in enumerate(idxs):
            color = hsv_color(idx,n_points)
            xL,yL = float(selL[j,0,0]), float(selL[j,0,1])
            xR,yR = float(selR[j,0,0]), float(selR[j,0,1])
            aR,bR,cR = linesR[j]; aL,bL,cL = linesL[j]
            cv2.circle(visL,(int(round(xL)),int(round(yL))),6,color,-1,cv2.LINE_AA)
            cv2.circle(visR,(int(round(xR)),int(round(yR))),6,color,-1,cv2.LINE_AA)
            draw_line(visR,(aR,bR,cR),color); draw_line(visL,(aL,bL,cL),color)
        cv2.imshow("LEFT Epipolar",visL)
        cv2.imshow("RIGHT Epipolar",visR)
        key = cv2.waitKey(0) & 0xFF
        if key==ord('q'): break
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
def main():
    # Calibration intrinsèque
    left_files = sorted(glob.glob(os.path.join(LEFT_DIR,"left_*.png")))
    right_files = sorted(glob.glob(os.path.join(RIGHT_DIR,"right_*.png")))
    K_L, dist_L, _, _, _ = calibrate_camera(left_files, "LEFT")
    K_R, dist_R, _, _, _ = calibrate_camera(right_files, "RIGHT")

    # Calibration stéréo extrinsèque
    stereo_extrinsic_calibration(K_L, dist_L, K_R, dist_R)

if __name__=="__main__":
    main()
import os

import time

import glob

import numpy as np

import cv2
 
# =========================

# PARAMÈTRES (LINUX)

# =========================

# D'après ton v4l2-ctl:

# HP HD Camera -> /dev/video0 /dev/video1

# UGREEN Camera -> /dev/video4 /dev/video5

CAM_LEFT  = "/dev/video4"   # HP (mets /dev/video4 si tu veux l'autre à gauche)

CAM_RIGHT = "/dev/video2"   # UGREEN
 
CHESSBOARD_SIZE = (9, 7)    # (colonnes, lignes) = coins INTERNES

SQUARE_SIZE = 0.015         # mètres (15 mm)
 
MIN_PAIRS = 10

OUT_DIR = "calib_dataset"

LEFT_DIR = os.path.join(OUT_DIR, "left")

RIGHT_DIR = os.path.join(OUT_DIR, "right")
 
CAPTURE_COOLDOWN = 0.4
 
# Capture settings (important pour latence/bandwidth)

CAP_W = 640

CAP_H = 480

CAP_FPS = 30
 
# =========================

# CRITÈRES / FLAGS OPENCV

# =========================

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

CALIB_CRITERIA  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
 
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
 
# =========================

# OUTILS

# =========================

def ensure_dirs():

    os.makedirs(LEFT_DIR, exist_ok=True)

    os.makedirs(RIGHT_DIR, exist_ok=True)
 
def build_object_points(chessboard_size, square_size):

    nx, ny = chessboard_size

    objp = np.zeros((nx * ny, 3), np.float32)

    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objp *= square_size

    return objp
 
def detect_corners(img, chessboard_size):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(gray, chessboard_size, FIND_FLAGS)

    if not ok:

        return False, None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)

    return True, corners
 
def list_pairs():

    lefts = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))

    pairs = []

    for lf in lefts:

        idx = os.path.basename(lf).replace("left_", "").replace(".png", "")

        rf = os.path.join(RIGHT_DIR, f"right_{idx}.png")

        if os.path.exists(rf):

            pairs.append((lf, rf))

    return pairs
 
def open_camera(dev_path: str):

    """Ouverture robuste (Linux V4L2) + réduction buffer + réglages."""

    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)

    if not cap.isOpened():

        raise RuntimeError(f"Impossible d'ouvrir {dev_path}")
 
    # Réduit la latence (buffer)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
    # Fixe même résolution/fps sur les deux caméras (réduit décalage + charge USB)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
 
    # Warm-up: vide les buffers

    for _ in range(20):

        cap.grab()
 
    # Vérifie lecture

    ok, frame = cap.read()

    if not ok or frame is None:

        cap.release()

        raise RuntimeError(f"{dev_path} s'ouvre mais ne renvoie pas d'images (read failed).")
 
    return cap
 
# =========================

# 1) CAPTURE SIMULTANÉE (latence réduite)

# =========================

def capture_pairs():

    ensure_dirs()
 
    capL = open_camera(CAM_LEFT)

    capR = open_camera(CAM_RIGHT)
 
    idx = len(list_pairs())

    last_shot = 0.0
 
    print("\n--- CAPTURE ---")

    print("Touches:")

    print("  c : capturer une paire (gauche+droite) (si damier détecté sur les deux)")

    print("  q : terminer la capture et passer à la calibration\n")
 
    while True:

        # grab quasi simultané (réduit le décalage)

        okgL = capL.grab()

        okgR = capR.grab()

        if not okgL or not okgR:

            print("Erreur grab caméra.")

            break
 
        okL, frameL = capL.retrieve()

        okR, frameR = capR.retrieve()

        if not okL or not okR or frameL is None or frameR is None:

            print("Erreur retrieve caméra.")

            break
 
        okCL, cornersL = detect_corners(frameL, CHESSBOARD_SIZE)

        okCR, cornersR = detect_corners(frameR, CHESSBOARD_SIZE)
 
        visL = frameL.copy()

        visR = frameR.copy()
 
        if okCL:

            cv2.drawChessboardCorners(visL, CHESSBOARD_SIZE, cornersL, okCL)

            cv2.putText(visL, "DAMIER OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        else:

            cv2.putText(visL, "damier non detecte", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
 
        if okCR:

            cv2.drawChessboardCorners(visR, CHESSBOARD_SIZE, cornersR, okCR)

            cv2.putText(visR, "DAMIER OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        else:

            cv2.putText(visR, "damier non detecte", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
 
        pairs_count = len(list_pairs())

        cv2.putText(visL, f"paires: {pairs_count}", (20, 80),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
 
        cv2.imshow("LEFT (capture)", visL)

        cv2.imshow("RIGHT (capture)", visR)
 
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            break
 
        if key == ord('c'):

            now = time.time()

            if now - last_shot < CAPTURE_COOLDOWN:

                continue
 
            # Strict : damier détecté sur les 2 images

            if not (okCL and okCR):

                print("Capture refusée : damier pas détecté sur les 2 caméras.")

                continue
 
            stamp = f"{idx:04d}"

            lf = os.path.join(LEFT_DIR, f"left_{stamp}.png")

            rf = os.path.join(RIGHT_DIR, f"right_{stamp}.png")
 
            cv2.imwrite(lf, frameL)

            cv2.imwrite(rf, frameR)

            print(f"[OK] Sauvé paire #{idx} : {lf} | {rf}")
 
            idx += 1

            last_shot = now
 
    capL.release()

    capR.release()

    cv2.destroyAllWindows()
 
# =========================

# 2) CALIBRATION (par caméra) + erreurs mean & RMS

# =========================

def calibrate_camera(image_files, chessboard_size, square_size, name="cam"):

    objp = build_object_points(chessboard_size, square_size)
 
    objpoints = []

    imgpoints = []

    img_size = None

    used_files = []
 
    for f in image_files:

        img = cv2.imread(f)

        if img is None:

            continue

        if img_size is None:

            img_size = (img.shape[1], img.shape[0])
 
        ok, corners = detect_corners(img, chessboard_size)

        if ok:

            objpoints.append(objp.copy())

            imgpoints.append(corners)

            used_files.append(f)
 
    if len(used_files) < 5:

        raise RuntimeError(f"[{name}] Pas assez d'images valides pour calibrer: {len(used_files)}.")
 
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(

        objpoints, imgpoints, img_size, None, None, criteria=CALIB_CRITERIA

    )
 
    # Erreurs reprojection: mean & RMS

    all_dists = []

    per_view_mean = []
 
    for i in range(len(objpoints)):

        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)

        detected = imgpoints[i].reshape(-1, 2)

        projected = proj.reshape(-1, 2)
 
        d = np.linalg.norm(detected - projected, axis=1)  # distances px

        all_dists.append(d)

        per_view_mean.append(float(np.mean(d)))
 
    all_dists = np.concatenate(all_dists)

    mean_err = float(np.mean(all_dists))

    rms_err = float(np.sqrt(np.mean(all_dists ** 2)))
 
    print(f"\n--- Calibration {name} ---")

    print(f"Images utilisées: {len(used_files)}/{len(image_files)}")

    print("K =\n", K)

    print("dist =", dist.ravel())

    print(f"Erreur moyenne reprojection: {mean_err:.4f} px")

    print(f"RMS reprojection:            {rms_err:.4f} px")
 
    return K, dist, rvecs, tvecs, used_files, mean_err, rms_err, per_view_mean
 
# =========================

# 3) VALIDATION VISUELLE

# =========================

def validate_reprojection(image_files, chessboard_size, square_size, K, dist, rvecs, tvecs, title):

    objp = build_object_points(chessboard_size, square_size)
 
    print(f"\n--- Validation reprojection {title} ---")

    print("Vert = coins détectés, Rouge = points reprojetés.")

    print("Touches: espace = image suivante | q = quitter")
 
    for i, f in enumerate(image_files):

        img = cv2.imread(f)

        if img is None:

            continue
 
        ok, corners = detect_corners(img, chessboard_size)

        if not ok:

            continue
 
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
 
        vis = img.copy()
 
        for p in corners.reshape(-1, 2):

            cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
 
        for p in proj.reshape(-1, 2):

            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
 
        cv2.putText(vis, f"{title} | view {i}", (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
 
        cv2.imshow(f"Validation - {title}", vis)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):

            break
 
    cv2.destroyAllWindows()
 
# =========================

# MAIN

# =========================

def main():

    capture_pairs()
 
    pairs = list_pairs()

    if len(pairs) < MIN_PAIRS:

        raise RuntimeError(f"Pas assez de paires ({len(pairs)}). Vise au moins {MIN_PAIRS}.")
 
    left_files  = [p[0] for p in pairs]

    right_files = [p[1] for p in pairs]
 
    K_L, dist_L, rvecs_L, tvecs_L, used_L, mean_L, rms_L, _ = calibrate_camera(

        left_files, CHESSBOARD_SIZE, SQUARE_SIZE, "LEFT"

    )

    K_R, dist_R, rvecs_R, tvecs_R, used_R, mean_R, rms_R, _ = calibrate_camera(

        right_files, CHESSBOARD_SIZE, SQUARE_SIZE, "RIGHT"

    )
 
    print("\n=========== RÉSULTATS FINAUX ===========")

    print("K_LEFT =\n", K_L)

    print("dist_LEFT =", dist_L.ravel())

    print(f"LEFT  mean err = {mean_L:.4f} px | RMS = {rms_L:.4f} px")

    print("\nK_RIGHT =\n", K_R)

    print("dist_RIGHT =", dist_R.ravel())

    print(f"RIGHT mean err = {mean_R:.4f} px | RMS = {rms_R:.4f} px")

    print("=======================================\n")
 
    np.savez(os.path.join(OUT_DIR, "intrinsics_left.npz"),

             K=K_L, dist=dist_L, mean_err=mean_L, rms_err=rms_L,

             rvecs=rvecs_L, tvecs=tvecs_L)

    np.savez(os.path.join(OUT_DIR, "intrinsics_right.npz"),

             K=K_R, dist=dist_R, mean_err=mean_R, rms_err=rms_R,

             rvecs=rvecs_R, tvecs=tvecs_R)
 
    print(f"Résultats sauvegardés dans {OUT_DIR}/intrinsics_left.npz et intrinsics_right.npz")
 
    validate_reprojection(used_L, CHESSBOARD_SIZE, SQUARE_SIZE, K_L, dist_L, rvecs_L, tvecs_L, "LEFT")

    validate_reprojection(used_R, CHESSBOARD_SIZE, SQUARE_SIZE, K_R, dist_R, rvecs_R, tvecs_R, "RIGHT")
 
 
if __name__ == "__main__":

    main()
 
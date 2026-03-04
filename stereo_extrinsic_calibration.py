import os
import glob
import random
import numpy as np
import cv2
 
# =========================
# CONFIG (à adapter)
# =========================
OUT_DIR = "calib_dataset"
LEFT_DIR = os.path.join(OUT_DIR, "left")
RIGHT_DIR = os.path.join(OUT_DIR, "right")
 
INTR_LEFT_PATH = os.path.join(OUT_DIR, "intrinsics_left.npz")
INTR_RIGHT_PATH = os.path.join(OUT_DIR, "intrinsics_right.npz")
 
CHESSBOARD_SIZE = (9, 7)   # coins internes (cols, rows)
SQUARE_SIZE = 0.015        # m
MIN_STEREO_PAIRS = 8
 
# Affichage épipolaire
N_RANDOM_POINTS = 5        # seulement 5 coins au hasard par paire
RANDOM_SEED = 42           # fixe pour reproductibilité (mets None pour vrai hasard)
 
# OpenCV
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
CALIB_CRITERIA  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
 
 
# =========================
# UTILITAIRES
# =========================
def list_pairs():
    lefts = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))
    pairs = []
    for lf in lefts:
        idx = os.path.basename(lf).replace("left_", "").replace(".png", "")
        rf = os.path.join(RIGHT_DIR, f"right_{idx}.png")
        if os.path.exists(rf):
            pairs.append((lf, rf))
    return pairs
 
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
 
def hsv_color(i, n):
    """Couleur stable par index (BGR)."""
    h = int((i / max(1, n - 1)) * 179)
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
 
def point_line_distance(x, y, a, b, c):
    return abs(a * x + b * y + c) / (np.sqrt(a * a + b * b) + 1e-12)
 
def draw_line(img, line_abc, color, thickness=2):
    """Trace ax+by+c=0 sur l'image."""
    a, b, c = line_abc
    h, w = img.shape[:2]
    eps = 1e-12
 
    if abs(b) > eps:
        y0 = int(round((-c - a * 0) / b))
        y1 = int(round((-c - a * (w - 1)) / b))
        pt1, pt2 = (0, y0), (w - 1, y1)
    else:
        x = int(round(-c / (a + eps)))
        pt1, pt2 = (x, 0), (x, h - 1)
 
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
 
 
# =========================
# CALIBRATION EXTRINSÈQUE + ÉPIPOLAIRES (5 points)
# =========================
def stereo_extrinsic_calibration():
    if not (os.path.exists(INTR_LEFT_PATH) and os.path.exists(INTR_RIGHT_PATH)):
        raise RuntimeError(
            "Intrinsèques manquants. Exécute d'abord la calibration intrinsèque et génère:\n"
            f"  - {INTR_LEFT_PATH}\n"
            f"  - {INTR_RIGHT_PATH}"
        )
 
    intrL = np.load(INTR_LEFT_PATH, allow_pickle=True)
    intrR = np.load(INTR_RIGHT_PATH, allow_pickle=True)
 
    K_L = intrL["K"]
    dist_L = intrL["dist"]
    K_R = intrR["K"]
    dist_R = intrR["dist"]
 
    pairs = list_pairs()
    if len(pairs) == 0:
        raise RuntimeError("Aucune paire trouvée dans calib_dataset/left et calib_dataset/right.")
 
    objp = build_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)
 
    objpoints = []
    imgpointsL = []
    imgpointsR = []
    used_pairs = []
    img_size = None
 
    # Collecte correspondances stéréo (damier visible sur les 2)
    for lf, rf in pairs:
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
        if imgL is None or imgR is None:
            continue
 
        if img_size is None:
            img_size = (imgL.shape[1], imgL.shape[0])
            print("K_L:", K_L)
            print("K_R:", K_R)
 
        okL, cornersL = detect_corners(imgL, CHESSBOARD_SIZE)
        okR, cornersR = detect_corners(imgR, CHESSBOARD_SIZE)
 
        if okL and okR:
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            used_pairs.append((lf, rf))
 
    if len(used_pairs) < MIN_STEREO_PAIRS:
        raise RuntimeError(f"Pas assez de paires stéréo valides: {len(used_pairs)} (min {MIN_STEREO_PAIRS}).")
 
    # Fixe les intrinsèques, estime uniquement extrinsèques + E/F
    flags = cv2.CALIB_FIX_INTRINSIC
 
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        K_L, dist_L, K_R, dist_R,
        img_size,
        criteria=CALIB_CRITERIA,
        flags=flags
    )
 
    print("\n=========== STEREO EXTRINSICS ===========")
    print(f"RMS stereoCalibrate: {rms:.6f}")
    print("\nR (rotation) =\n", R)
    print("\nT (translation) =\n", T)
    print("\nE =\n", E)
    print("\nF =\n", F)
    print("========================================\n")
 
    # Sauvegarde
    out_path = os.path.join(OUT_DIR, "stereo_extrinsics.npz")
    np.savez(out_path, R=R, T=T, E=E, F=F, rms=rms,
             K_L=K_L, dist_L=dist_L, K_R=K_R, dist_R=dist_R,
             chessboard_size=np.array(CHESSBOARD_SIZE), square_size=float(SQUARE_SIZE),
             used_pairs=np.array(used_pairs, dtype=object))
    print(f"Sauvegardé: {out_path}\n")
 
    # Visualisation épipolaire sur quelques paires
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
 
    n_points = CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]
 
    print("--- Visualisation épipolaire (5 points aléatoires) ---")
    print("Même couleur = point et sa ligne épipolaire.")
    print("Touches: espace = paire suivante | q = quitter\n")
 
    all_sym_err = []
 
    for i, (lf, rf) in enumerate(used_pairs):
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
 
        okL, cornersL = detect_corners(imgL, CHESSBOARD_SIZE)
        okR, cornersR = detect_corners(imgR, CHESSBOARD_SIZE)
        if not (okL and okR):
            continue
 
        ptsL = cornersL.reshape(-1, 1, 2).astype(np.float32)
        ptsR = cornersR.reshape(-1, 1, 2).astype(np.float32)
 
        # Choix de 5 indices au hasard
        idxs = random.sample(range(n_points), k=min(N_RANDOM_POINTS, n_points))
 
        # Lignes épipolaires pour ces points
        selL = ptsL[idxs]  # points gauche choisis
        selR = ptsR[idxs]  # points droite correspondants
 
        linesR = cv2.computeCorrespondEpilines(selL, 1, F).reshape(-1, 3)
        linesL = cv2.computeCorrespondEpilines(selR, 2, F).reshape(-1, 3)
 
        visL = imgL.copy()
        visR = imgR.copy()
 
        sym_errs_this = []
 
        for j, idx in enumerate(idxs):
            color = hsv_color(idx, n_points)
 
            xL, yL = float(selL[j, 0, 0]), float(selL[j, 0, 1])
            xR, yR = float(selR[j, 0, 0]), float(selR[j, 0, 1])
 
            aR, bR, cR = linesR[j]
            aL, bL, cL = linesL[j]
 
            # erreurs point-ligne (px)
            err_L2R = point_line_distance(xR, yR, aR, bR, cR)
            err_R2L = point_line_distance(xL, yL, aL, bL, cL)
            err_sym = 0.5 * (err_L2R + err_R2L)
            sym_errs_this.append(err_sym)
 
            # dessin points
            cv2.circle(visL, (int(round(xL)), int(round(yL))), 6, color, -1, cv2.LINE_AA)
            cv2.circle(visR, (int(round(xR)), int(round(yR))), 6, color, -1, cv2.LINE_AA)
 
            # dessin lignes
            draw_line(visR, (aR, bR, cR), color, thickness=2)
            draw_line(visL, (aL, bL, cL), color, thickness=2)
 
            # annotation courte par point
            cv2.putText(visR, f"{idx}:{err_L2R:.2f}px",
                        (int(round(xR)) + 8, int(round(yR)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(visL, f"{idx}:{err_R2L:.2f}px",
                        (int(round(xL)) + 8, int(round(yL)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
        sym_errs_this = np.array(sym_errs_this, dtype=np.float64)
        all_sym_err.append(sym_errs_this)
 
        mean_sym = float(np.mean(sym_errs_this))
        max_sym = float(np.max(sym_errs_this))
 
        cv2.putText(visL, f"pair {i} | mean epi err: {mean_sym:.3f}px | max: {max_sym:.3f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(visR, f"pair {i} | mean epi err: {mean_sym:.3f}px | max: {max_sym:.3f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
 
        cv2.imshow("LEFT - epipolar (5 points)", visL)
        cv2.imshow("RIGHT - epipolar (5 points)", visR)
 
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
 
    cv2.destroyAllWindows()
 
    all_sym_err = np.concatenate(all_sym_err, axis=0)
    print("\n=========== EPIPOLAR ERROR (5 points/pair) ===========")
    print(f"Mean:   {np.mean(all_sym_err):.6f} px")
    print(f"Median: {np.median(all_sym_err):.6f} px")
    print(f"Std:    {np.std(all_sym_err):.6f} px")
    print(f"Max:    {np.max(all_sym_err):.6f} px")
    print("======================================================\n")
 
 
if __name__ == "__main__":
    stereo_extrinsic_calibration()
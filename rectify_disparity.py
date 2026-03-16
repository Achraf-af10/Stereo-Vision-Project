import os
import glob
import cv2
import numpy as np
 
# =========================
# CONFIG
# =========================
BASE_DIR = "test_disparity"
 
LEFT_DIR = os.path.join(BASE_DIR, "left")
RIGHT_DIR = os.path.join(BASE_DIR, "right")
 
RECTIFIED_DIR = os.path.join(BASE_DIR, "rectified")
RECT_LEFT_DIR = os.path.join(RECTIFIED_DIR, "left")
RECT_RIGHT_DIR = os.path.join(RECTIFIED_DIR, "right")
 
INTR_LEFT_PATH = os.path.join(BASE_DIR, "intrinsics_left.npz")
INTR_RIGHT_PATH = os.path.join(BASE_DIR, "intrinsics_right.npz")
EXTR_PATH = os.path.join(BASE_DIR, "stereo_extrinsics.npz")
 
ALPHA = 0
SHOW_ALL_PAIRS = True
 
# =========================
# OUTILS
# =========================
def ensure_dirs():
    os.makedirs(RECT_LEFT_DIR, exist_ok=True)
    os.makedirs(RECT_RIGHT_DIR, exist_ok=True)
 
def list_pairs():
    left_images = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
    pairs = []
 
    for lf in left_images:
        name = os.path.basename(lf)
        idx = name.replace("Im_L_", "").replace(".png", "")
 
        rf = os.path.join(RIGHT_DIR, f"Im_R_{idx}.png")
 
        if os.path.exists(rf):
            pairs.append((lf, rf, idx))
 
    return pairs
 
def draw_horizontal_lines(img, step=40):
    vis = img.copy()
    h, w = vis.shape[:2]
 
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w - 1, y), (0,255,0), 1)
 
    return vis
 
def stack_images(img1, img2):
    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Images de hauteurs différentes")
 
    return np.hstack((img1, img2))
 
# =========================
# RECTIFICATION
# =========================
def main():
 
    ensure_dirs()
 
    # -------------------------
    # Vérifier fichiers calibration
    # -------------------------
    if not os.path.exists(INTR_LEFT_PATH):
        raise RuntimeError(f"Intrinsics left manquant : {INTR_LEFT_PATH}")
 
    if not os.path.exists(INTR_RIGHT_PATH):
        raise RuntimeError(f"Intrinsics right manquant : {INTR_RIGHT_PATH}")
 
    if not os.path.exists(EXTR_PATH):
        raise RuntimeError(f"Extrinsics manquant : {EXTR_PATH}")
 
    # -------------------------
    # Charger calibration
    # -------------------------
    intrL = np.load(INTR_LEFT_PATH)
    intrR = np.load(INTR_RIGHT_PATH)
 
    K_L = intrL["K"]
    dist_L = intrL["dist"]
 
    K_R = intrR["K"]
    dist_R = intrR["dist"]
 
    extr = np.load(EXTR_PATH)
 
    R = extr["R"]
    T = extr["T"]
 
    print("=== PARAMETRES ===")
    print("K_L\n", K_L)
    print("K_R\n", K_R)
 
    # -------------------------
    # Charger paires
    # -------------------------
    pairs = list_pairs()
 
    if len(pairs) == 0:
        raise RuntimeError("Aucune paire trouvée dans les dossiers left/right")
 
    print("Nombre de paires trouvées :", len(pairs))
 
    # -------------------------
    # Taille image
    # -------------------------
    imgL0 = cv2.imread(pairs[0][0])
    imgR0 = cv2.imread(pairs[0][1])
 
    if imgL0 is None or imgR0 is None:
        raise RuntimeError("Impossible de lire la première paire")
 
    if imgL0.shape[:2] != imgR0.shape[:2]:
        raise RuntimeError("Les tailles d'images ne correspondent pas")
 
    image_size = (imgL0.shape[1], imgL0.shape[0])
    print("Image size :", image_size)
 
    # -------------------------
    # Calcul rectification
    # -------------------------
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_L, dist_L,
        K_R, dist_R,
        image_size,
        R, T,
        alpha=ALPHA
    )
 
    # -------------------------
    # Maps rectification
    # -------------------------
    map1x, map1y = cv2.initUndistortRectifyMap(
        K_L, dist_L, R1, P1, image_size, cv2.CV_32FC1
    )
 
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_R, dist_R, R2, P2, image_size, cv2.CV_32FC1
    )
 
    # -------------------------
    # Sauvegarder paramètres
    # -------------------------
    np.savez(
        os.path.join(BASE_DIR, "rectification_maps.npz"),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q
    )
 
    # -------------------------
    # Rectifier toutes les images
    # -------------------------
    for i,(lf,rf,idx) in enumerate(pairs):
 
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
 
        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
 
        # sauvegarde
        outL = os.path.join(RECT_LEFT_DIR, f"rect_left_{idx}.png")
        outR = os.path.join(RECT_RIGHT_DIR, f"rect_right_{idx}.png")
 
        cv2.imwrite(outL, rectL)
        cv2.imwrite(outR, rectR)
 
        # affichage
        rectL_lines = draw_horizontal_lines(rectL)
        rectR_lines = draw_horizontal_lines(rectR)
 
        stacked = stack_images(rectL_lines, rectR_lines)
 
        cv2.imshow("Rectification", stacked)
 
        key = cv2.waitKey(0)
 
        if key == ord('q'):
            break
 
        if not SHOW_ALL_PAIRS:
            break
 
    cv2.destroyAllWindows()
 
    print("\nImages rectifiées enregistrées dans :")
    print(RECT_LEFT_DIR)
    print(RECT_RIGHT_DIR)
 
if __name__ == "__main__":
    main()
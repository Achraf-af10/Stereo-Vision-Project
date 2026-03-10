import os
import glob
import cv2
import numpy as np
 
# =========================
# CONFIG
# =========================
OUT_DIR = "calib_dataset"
LEFT_DIR = os.path.join(OUT_DIR, "left")
RIGHT_DIR = os.path.join(OUT_DIR, "right")
 
INTR_LEFT_PATH = os.path.join(OUT_DIR, "intrinsics_left.npz")
INTR_RIGHT_PATH = os.path.join(OUT_DIR, "intrinsics_right.npz")
EXTR_PATH = os.path.join(OUT_DIR, "stereo_extrinsics.npz")
 
RECTIFIED_DIR = os.path.join(OUT_DIR, "rectified")
RECT_LEFT_DIR = os.path.join(RECTIFIED_DIR, "left")
RECT_RIGHT_DIR = os.path.join(RECTIFIED_DIR, "right")
 
# 0 = zoom possible / coupe des bords noirs
# 1 = garde tous les pixels / plus de bords noirs
ALPHA = 0
 
# Afficher seulement une paire ou toutes les paires
SHOW_ALL_PAIRS = True
 
# =========================
# OUTILS
# =========================
def ensure_dirs():
    os.makedirs(RECT_LEFT_DIR, exist_ok=True)
    os.makedirs(RECT_RIGHT_DIR, exist_ok=True)
 
def list_pairs():
    lefts = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))
    pairs = []
 
    for lf in lefts:
        idx = os.path.basename(lf).replace("left_", "").replace(".png", "")
        rf = os.path.join(RIGHT_DIR, f"right_{idx}.png")
 
        if os.path.exists(rf):
            pairs.append((lf, rf, idx))
 
    return pairs
 
def draw_horizontal_lines(img, step=40, color=(0, 255, 0)):
    vis = img.copy()
    h, w = vis.shape[:2]
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
    return vis
 
def stack_images_horizontally(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
 
    if h1 != h2:
        raise ValueError("Les deux images n'ont pas la même hauteur.")
 
    return np.hstack((img1, img2))
 
# =========================
# RECTIFICATION
# =========================
def main():
    ensure_dirs()
 
    # Vérification fichiers calibration
    if not os.path.exists(INTR_LEFT_PATH):
        raise RuntimeError(f"Fichier manquant: {INTR_LEFT_PATH}")
    if not os.path.exists(INTR_RIGHT_PATH):
        raise RuntimeError(f"Fichier manquant: {INTR_RIGHT_PATH}")
    if not os.path.exists(EXTR_PATH):
        raise RuntimeError(f"Fichier manquant: {EXTR_PATH}")
 
    # Chargement intrinsèques
    intrL = np.load(INTR_LEFT_PATH, allow_pickle=True)
    intrR = np.load(INTR_RIGHT_PATH, allow_pickle=True)
 
    K_L = intrL["K"]
    dist_L = intrL["dist"]
 
    K_R = intrR["K"]
    dist_R = intrR["dist"]
 
    # Chargement extrinsèques
    extr = np.load(EXTR_PATH, allow_pickle=True)
    R = extr["R"]
    T = extr["T"]
 
    print("=== PARAMÈTRES CHARGÉS ===")
    print("K_L =\n", K_L)
    print("dist_L =", dist_L.ravel())
    print("\nK_R =\n", K_R)
    print("dist_R =", dist_R.ravel())
    print("\nR =\n", R)
    print("\nT =\n", T)
 
    # Liste des paires
    pairs = list_pairs()
    if len(pairs) == 0:
        raise RuntimeError("Aucune paire d'images trouvée dans calib_dataset/left et calib_dataset/right.")
 
    # Lire la première paire pour récupérer la taille image
    imgL0 = cv2.imread(pairs[0][0])
    imgR0 = cv2.imread(pairs[0][1])
 
    if imgL0 is None or imgR0 is None:
        raise RuntimeError("Impossible de lire la première paire d'images.")
 
    if imgL0.shape[:2] != imgR0.shape[:2]:
        raise RuntimeError(
            f"Les tailles diffèrent: left={imgL0.shape[:2]}, right={imgR0.shape[:2]}"
        )
 
    image_size = (imgL0.shape[1], imgL0.shape[0])  # (width, height)
    print("\nimage_size =", image_size)
 
    # Calcul rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_L, dist_L,
        K_R, dist_R,
        image_size,
        R, T,
        alpha=ALPHA
    )
 
    print("\n=== RECTIFICATION ===")
    print("R1 =\n", R1)
    print("\nR2 =\n", R2)
    print("\nP1 =\n", P1)
    print("\nP2 =\n", P2)
    print("\nQ =\n", Q)
    print("\nroi1 =", roi1)
    print("roi2 =", roi2)
 
    # Maps de rectification
    map1x, map1y = cv2.initUndistortRectifyMap(
        K_L, dist_L, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_R, dist_R, R2, P2, image_size, cv2.CV_32FC1
    )
 
    # Sauvegarder aussi les paramètres de rectification
    np.savez(
        os.path.join(OUT_DIR, "rectification_maps.npz"),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=np.array(roi1), roi2=np.array(roi2)
    )
 
    print("\nParamètres de rectification sauvegardés dans calib_dataset/rectification_maps.npz")
 
    # Parcours des paires
    for i, (lf, rf, idx) in enumerate(pairs):
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
 
        if imgL is None or imgR is None:
            continue
 
        rectL = cv2.remap(imgL, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, interpolation=cv2.INTER_LINEAR)
 
        # Sauvegarde
        outL = os.path.join(RECT_LEFT_DIR, f"rect_left_{idx}.png")
        outR = os.path.join(RECT_RIGHT_DIR, f"rect_right_{idx}.png")
        cv2.imwrite(outL, rectL)
        cv2.imwrite(outR, rectR)
 
        # Vérification visuelle avec lignes horizontales
        rectL_lines = draw_horizontal_lines(rectL, step=40, color=(0, 255, 0))
        rectR_lines = draw_horizontal_lines(rectR, step=40, color=(0, 255, 0))
 
        stacked = stack_images_horizontally(rectL_lines, rectR_lines)
        cv2.putText(
            stacked,
            f"Pair {i} / index {idx} - lignes horizontales = verif rectification",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
 
        cv2.imshow("Stereo Rectification", stacked)
 
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
 
        if not SHOW_ALL_PAIRS:
            break
 
    cv2.destroyAllWindows()
    print("\nImages rectifiées sauvegardées dans :")
    print(" -", RECT_LEFT_DIR)
    print(" -", RECT_RIGHT_DIR)
 
 
if __name__ == "__main__":
    main()
import os
import time
import glob
import cv2

# =========================
# PARAMÈTRES
# =========================
CAM_LEFT  = "/dev/video2"
CAM_RIGHT = "/dev/video4"

OUT_DIR = "calib_dataset"
LEFT_DIR = os.path.join(OUT_DIR, "left")
RIGHT_DIR = os.path.join(OUT_DIR, "right")

CAPTURE_COOLDOWN = 0.4
CAP_W, CAP_H, CAP_FPS = 1024, 576, 15

CHESSBOARD_SIZE = (11, 7)
SQUARE_SIZE = 0.0265  # mètres (15 mm)

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# =========================
# OUTILS
# =========================
def ensure_dirs():
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, FIND_FLAGS)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return ok, corners

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
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {dev_path}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    for _ in range(20): cap.grab()
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"{dev_path} s'ouvre mais ne renvoie pas d'images.")
    return cap

# =========================
# CAPTURE
# =========================
def capture_pairs():
    ensure_dirs()
    capL, capR = open_camera(CAM_LEFT), open_camera(CAM_RIGHT)
    idx, last_shot = len(list_pairs()), 0.0

    print("\n--- CAPTURE ---")
    print("c : capturer une paire | q : quitter")

    while True:
        if not capL.grab() or not capR.grab():
            print("Erreur grab")
            break
        okL, frameL = capL.retrieve()
        okR, frameR = capR.retrieve()
        if not okL or not okR or frameL is None or frameR is None:
            print("Erreur retrieve")
            break

        okCL, cornersL = detect_corners(frameL)
        okCR, cornersR = detect_corners(frameR)

        visL, visR = frameL.copy(), frameR.copy()
        cv2.putText(visL, "DAMIER OK" if okCL else "damier non detecte", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if okCL else (0,0,255),2)
        cv2.putText(visR, "DAMIER OK" if okCR else "damier non detecte", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if okCR else (0,0,255),2)
        cv2.imshow("LEFT", visL)
        cv2.imshow("RIGHT", visR)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            now = time.time()
            if now - last_shot < CAPTURE_COOLDOWN: continue
            if not (okCL and okCR):
                print("Capture refusée : damier pas détecté sur les 2 caméras.")
                continue
            lf = os.path.join(LEFT_DIR, f"left_{idx:04d}.png")
            rf = os.path.join(RIGHT_DIR, f"right_{idx:04d}.png")
            cv2.imwrite(lf, frameL)
            cv2.imwrite(rf, frameR)
            print(f"[OK] Sauvé paire #{idx}")
            idx += 1
            last_shot = now

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    capture_pairs()
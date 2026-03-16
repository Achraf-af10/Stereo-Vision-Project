import os

import time

import cv2
 
# =========================

# PARAMÈTRES

# =========================


CAM_LEFT = "/dev/video2"

CAM_RIGHT = "/dev/video4"
 
SAVE_DIR = "test_disparity"

LEFT_DIR = os.path.join(SAVE_DIR, "left")

RIGHT_DIR = os.path.join(SAVE_DIR, "right")
 
CAP_W = 1024

CAP_H = 576

CAP_FPS = 15
 
CAPTURE_COOLDOWN = 0.4 
 
# =========================

# OUTILS

# =========================

def ensure_dirs():

    os.makedirs(LEFT_DIR, exist_ok=True)

    os.makedirs(RIGHT_DIR, exist_ok=True)
 
def open_camera(device):

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
 
    if not cap.isOpened():

        raise RuntimeError(f"Impossible d'ouvrir la caméra {device}")
 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
 
    # warm-up

    for _ in range(10):

        cap.grab()
 
    ok, frame = cap.read()

    if not ok or frame is None:

        cap.release()

        raise RuntimeError(f"La caméra {device} s'ouvre mais ne renvoie pas d'image")
 
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[OK] Caméra {device} ouverte | résolution réelle = {real_w}x{real_h}")
 
    return cap
 
def next_index():

    existing = []

    for folder, prefix in [(LEFT_DIR, "Im_L_"), (RIGHT_DIR, "Im_R_")]:

        for name in os.listdir(folder):

            if name.startswith(prefix) and name.endswith(".png"):

                try:

                    idx = int(name.replace(prefix, "").replace(".png", ""))

                    existing.append(idx)

                except ValueError:

                    pass

    return 1 if not existing else max(existing) + 1
 
# =========================

# CAPTURE STÉRÉO

# =========================

def main():

    ensure_dirs()
 
    print("Ouverture caméra gauche...")

    capL = open_camera(CAM_LEFT)
 
    print("Ouverture caméra droite...")

    capR = open_camera(CAM_RIGHT)
 
    idx = next_index()

    last_capture = 0.0
 
    print("\n--- CAPTURE STÉRÉO TEST DISPARITÉ ---")

    print("Touches :")

    print("  c : capturer une paire")

    print("  q : quitter\n")
 
    while True:

        # grab quasi simultané

        okgL = capL.grab()

        okgR = capR.grab()
 
        if not okgL or not okgR:

            print("Erreur grab sur une caméra.")

            break
 
        okL, frameL = capL.retrieve()

        okR, frameR = capR.retrieve()
 
        if not okL or not okR or frameL is None or frameR is None:

            print("Erreur retrieve sur une caméra.")

            break
 
        visL = frameL.copy()

        visR = frameR.copy()
 
        cv2.putText(visL, f"LEFT | index prochain: {idx}", (20, 35),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(visR, f"RIGHT | index prochain: {idx}", (20, 35),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 
        cv2.imshow("LEFT", visL)

        cv2.imshow("RIGHT", visR)
 
        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('q'):

            break
 
        if key == ord('c'):

            now = time.time()

            if now - last_capture < CAPTURE_COOLDOWN:

                continue
 
            left_path = os.path.join(LEFT_DIR, f"Im_L_{idx}.png")

            right_path = os.path.join(RIGHT_DIR, f"Im_R_{idx}.png")
 
            cv2.imwrite(left_path, frameL)

            cv2.imwrite(right_path, frameR)
 
            print(f"[OK] Paire sauvegardée #{idx}")

            print(f"     {left_path}")

            print(f"     {right_path}")
 
            idx += 1

            last_capture = now
 
    capL.release()

    capR.release()

    cv2.destroyAllWindows()
 
if __name__ == "__main__":

    main()
 
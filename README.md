<div align="center">

# 🎯 Stereo Vision Project

**Full stereo vision pipeline — from camera calibration to dense 3D reconstruction**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-0.18-FF6F00?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-stable-brightgreen?style=for-the-badge)

</div>

---

## 📖 Overview

This project implements a **complete stereo vision pipeline** using two synchronized cameras and a chessboard calibration target. Starting from raw stereo image pairs, it covers every stage of the 3D perception chain:

1. **Intrinsic calibration** of each camera individually
2. **Extrinsic (stereo) calibration** to recover the relative pose between the two cameras
3. **Stereo rectification** to align epipolar lines horizontally
4. **Disparity map computation** using the StereoSGBM algorithm
5. **3D point cloud reconstruction** with Open3D

> Developed as part of a Computer Vision course at **Polytech Dijon – Le Creusot** (Robotics Engineering, Year 2), supervised by **Yohan Fougerolle**.

---

## 🗂️ Project Structure

```
Stereo-Vision-Project/
│
├── calib_dataset/              # Stereo image pairs used for calibration (chessboard)
├── test_disparity/             # Test image pairs for disparity & 3D reconstruction
│
├── stereo_capture_calib.py     # Capture calibration image pairs from live cameras
├── stereo_capture_disparity.py # Capture test image pairs for disparity estimation
│
├── calib_intrinsic.py          # Per-camera intrinsic calibration (camera matrix + distortion)
├── calib_extrinsic.py          # Stereo extrinsic calibration (R, T, E, F matrices)
├── rectify_calib.py            # Compute & save rectification maps (stereoRectify)
├── rectify_disparity.py        # Apply rectification to test image pairs
├── disparity.py                # StereoSGBM disparity map + Open3D 3D reconstruction
│
└── venv/                       # Python virtual environment (not tracked)
```

---

## 🔄 Pipeline Architecture

```
[Left Camera]  ──┐
                 ├──► Intrinsic Calibration ──► Stereo Calibration ──► Rectification
[Right Camera] ──┘                                                          │
                                                                            ▼
                                                               Rectified Stereo Pair
                                                                            │
                                                                            ▼
                                                               StereoSGBM Disparity Map
                                                                            │
                                                                            ▼
                                                               3D Point Cloud (Open3D)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Two USB cameras (or a stereo camera rig)
- A printed chessboard calibration pattern

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Achraf-af10/Stereo-Vision-Project.git
cd Stereo-Vision-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install opencv-python numpy open3d
```

### Running the Pipeline

```bash
# Step 1 — Capture calibration image pairs (press SPACE to save, Q to quit)
python stereo_capture_calib.py

# Step 2 — Compute intrinsic parameters for each camera
python calib_intrinsic.py

# Step 3 — Compute extrinsic (stereo) calibration
python calib_extrinsic.py

# Step 4 — Compute rectification maps
python rectify_calib.py

# Step 5 — Capture test pairs for disparity
python stereo_capture_disparity.py

# Step 6 — Rectify test pairs and compute disparity + 3D reconstruction
python rectify_disparity.py
python disparity.py
```

---

## 📸 Demo

> *(Add your own screenshots or GIFs here — disparity map, point cloud visualization, rectified pairs, etc.)*

| Rectified Pair | Disparity Map | 3D Point Cloud |
|:-:|:-:|:-:|
| ![rectified](https://via.placeholder.com/220x140?text=Rectified+Pair) | ![disparity](https://via.placeholder.com/220x140?text=Disparity+Map) | ![pointcloud](https://via.placeholder.com/220x140?text=3D+Point+Cloud) |

---

## ⚙️ Key Parameters

| Parameter | Location | Description |
|---|---|---|
| `CHESSBOARD_SIZE` | `calib_intrinsic.py` | Inner corners of the calibration pattern (e.g. `(9, 6)`) |
| `SQUARE_SIZE` | `calib_intrinsic.py` | Physical size of one square in mm |
| `numDisparities` | `disparity.py` | StereoSGBM — must be a multiple of 16 |
| `blockSize` | `disparity.py` | StereoSGBM matching block size (odd number) |
| Camera indices | `stereo_capture_*.py` | Adjust `cv2.VideoCapture(0/1)` to match your setup |

---

## 🛠️ Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` | Camera capture, calibration, rectification, StereoSGBM |
| `numpy` | Matrix operations and data handling |
| `open3d` | 3D point cloud generation and visualization |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "feat: add your feature"`
4. **Push** to your branch: `git push origin feature/your-feature`
5. Open a **Pull Request**

Please keep your code clean, add comments where necessary, and test your changes before submitting.

---

## 👤 Author

**Achraf Ahmed Fouatih**
Engineering Student — Robotics Specialty, Polytech Dijon (Le Creusot)

[![GitHub](https://img.shields.io/badge/GitHub-Achraf--af10-181717?style=flat&logo=github)](https://github.com/Achraf-af10)

---

<div align="center">
<sub>Built with 🔬 and OpenCV · Polytech Dijon 2025</sub>
</div>

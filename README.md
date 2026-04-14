<div align="center">

# 🎯 Stereo Vision Pipeline for 3D Reconstruction

**Full stereo vision pipeline — from camera calibration to dense 3D reconstruction**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-0.18-FF6F00?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-stable-brightgreen?style=for-the-badge)

</div>

---

## 📖 Overview

This project implements a **complete stereo vision pipeline** using two synchronized cameras and a chessboard calibration target. It covers the full 3D perception chain:

1. Intrinsic calibration of each camera  
2. Stereo (extrinsic) calibration  
3. Stereo rectification  
4. Disparity map computation (StereoSGBM)  
5. 3D point cloud reconstruction (Open3D)

> Developed at **Polytech Dijon – Le Creusot** (Robotics Engineering, Year 2).

---

## 📷 Hardware Setup

- 2 × Logitech C270 HD Webcam (720p)
- Resolution: 1280×720
- Fixed stereo rig setup
- Printed chessboard calibration pattern

---

## 🗂️ Project Structure

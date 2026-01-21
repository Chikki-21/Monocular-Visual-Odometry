# Monocular Visual Odometry (OpenCV)

This project implements a **monocular visual odometry pipeline** using
ORB feature detection, Lucas–Kanade optical flow, and Essential Matrix
pose estimation.

The system is implemented in an **object-oriented design**, focusing on
robust state handling and failure recovery.

---

## Pipeline Overview
1. Frame acquisition
2. Feature detection (ORB)
3. Feature tracking (Lucas–Kanade)
4. Essential matrix estimation
5. Pose recovery
6. Trajectory accumulation

---

## Key Concepts Used
- Monocular geometry
- Feature-based tracking
- Epipolar constraints
- Camera intrinsics
- State-based OOP design

---

## How to Run
```bash
pip install -r requirements.txt
python src/visual_odometry.py


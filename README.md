# CS436 Structure from Motion Project (Group 17)

**Muhammad Shazil Nadeem (27100183)**  
**Muhammad Taimur Jahanzeb (27100028)**  
Group 17

This repository contains our implementation for the CS436 Computer Vision Fundamentals semester project. The project is implemented up to **Week 3**, including feature matching, two-view geometry, and incremental multi-view Structure from Motion.

The code is modular and organized into scripts for each week and reusable components under the `src/` directory.

---

## Project Structure

```
CS436-SFM-PROJECT-GROUP17/
│
├── data/
│   ├── images/                 # Input images
│   └── results/
│       ├── week1/
│       ├── week2/
│       └── week3/
│
├── notebooks/
│   ├── week3_g17.ipynb          # Notebook version of Week 3
│   └── outputs/
│       ├── fountain_sfm_week3.ply
│       └── point_cloud.ply
│
├── scripts/
│   ├── run_week1_feature_matching.py
│   ├── run_week2_two_view.py
│   └── run_week3_multiview.py
│
├── src/
│   ├── features.py
│   ├── io_utils.py
│   ├── multiview_sfm.py
│   ├── two_view.py
│   └── vis_open3d.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create a Python environment (recommended: conda)


conda create -n cs436_sfm python=3.10
conda activate cs436_sfm



### 2. Install dependencies


pip install -r requirements.txt



### 3. If images are in HEIC format (iPhone)


pip install pillow-heif



### 4. Place all project images inside:


data/images/


---

## Running Each Week

All scripts must be executed **from the project root directory**.

---

### Week 1: Feature Matching


python -m scripts.run_week1_feature_matching


Outputs saved in:

data/results/week1/


---

### Week 2: Two-View Reconstruction

python -m scripts.run_week2_two_view


Outputs saved in:

data/results/week2/



These include:

- 3D point cloud (PLY)
- 3D Matplotlib visualization
- x–y projection visualization

---

### Week 3: Multi-View Incremental SfM

python -m scripts.run_week3_multiview


Outputs saved in:

data/results/week3/



These include:

- Multi-view point cloud (PLY)
- 3D Matplotlib visualization
- x–y projection visualization

---

## Summary

This repository contains all code developed up to **Week 3** of the CS436 SFM project.  
The implementation includes:

- SIFT feature extraction and matching  
- Essential matrix estimation and two-view pose recovery  
- Triangulation and reprojection filtering  
- Incremental camera registration using PnP  
- Multi-view 3D point cloud generation  
- Open3D visualization with consistent colors and black background  

All code is modular and follows a clear structure that separates runnable scripts from reusable modules.

---

